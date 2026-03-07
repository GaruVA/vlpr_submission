"""
src/spatial.py
Layer 3 — Spatial-Temporal Correlation (STC) Engine

This module fulfils three distinct academic claims from the Interim Report
and must be demonstrable during the viva:

  1. FacilityGraph (Interim Report §5.2 Class Diagram, §6.4 Algorithm 2):
     "The dockyard is modelled as a directed graph G = (V, E) where vertices
     represent gate camera nodes and edges represent physical road connections."
     Implemented as a pure-Python adjacency dictionary — no external libraries —
     to demonstrate explicit command of fundamental graph data structures.

  2. FraudAlert (Interim Report §5.2 Class Diagram):
     Structured data class with fields plate_number, violation_type,
     severity_level, delta_time as specified in the class diagram.

  3. SpatialTemporalVerifier (Interim Report §5.2, §6.4):
     "Each time a vehicle is detected at a gate, the engine retrieves the
     vehicle's previous gate and the elapsed time since that detection. If the
     elapsed time is less than the edge weight for that route, the system
     raises a fraud alert."

Sprint 2 Additions:
  - EDGE-001: TTL-based eviction of stale vehicle_state records.
  - EDGE-002: Guard against negative ΔT from NTP/clock adjustments.
  - FacilityGraph replaces the raw travel_times dict as the graph model.
  - FraudAlert dataclass replaces anonymous dicts in fraud_alerts list.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# FraudAlert — Structured Fraud Event Record
# Interim Report §5.2 Class Diagram: FraudAlert class with fields
#   +plate_number: String
#   +violation_type: String
#   +severity_level: String
#   +delta_time: Float
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FraudAlert:
    """
    Immutable record of a detected spatial-temporal fraud event.

    Directly implements the FraudAlert class from the system class diagram
    (Interim Report Figure 5.2). Each instance corresponds to one row in
    the SQLite fraud_events table (Interim Report §5.3 ER Diagram).

    Violation type semantics (Interim Report §6.4):
      CLONE_ATTACK      — ΔT < 0.5s: physically impossible. Two vehicles
                          cannot share one identity; one must be a clone.
      SPEEDING_VIOLATION — 0.5s ≤ ΔT < min_time: movement faster than the
                           speed limit allows; credential sharing suspected.
      PATH_VIOLATION    — No direct edge exists in the FacilityGraph for the
                          observed gate transition; checkpoint was skipped.
    """
    timestamp:      str    # ISO-format string, millisecond precision
    plate_number:   str    # Normalised Sri Lankan plate (e.g. "CAB-1234")
    origin_gate:    str    # Gate where the vehicle was last confirmed
    dest_gate:      str    # Gate where the impossible detection occurred
    violation_type: str    # 'CLONE_ATTACK' | 'SPEEDING_VIOLATION' | 'PATH_VIOLATION'
    severity_level: str    # 'CRITICAL' | 'HIGH' | 'MEDIUM'
    min_time:       float  # Minimum physical travel time (seconds)
    delta_time:     float  # Actual observed ΔT (seconds)
    reason:         str    # Human-readable description for the dashboard

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export / audit log display."""
        return {
            'timestamp':      self.timestamp,
            'plate':          self.plate_number,
            'origin_gate':    self.origin_gate,
            'dest_gate':      self.dest_gate,
            'violation_type': self.violation_type,
            'severity_level': self.severity_level,
            'min_time_s':     self.min_time,
            'delta_time_s':   round(self.delta_time, 3),
            'reason':         self.reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FacilityGraph — Pure-Python Directed Graph of the Dockyard
# Interim Report §5.2 Class Diagram, §6.4 Algorithm 2:
#   -nodes: List
#   -edges: List
#   +get_travel_time(source_gate, dest_gate): Float
# ─────────────────────────────────────────────────────────────────────────────

class FacilityGraph:
    """
    Directed graph model of the Colombo Dockyard facility.

    Implements the FacilityGraph class described in Interim Report §5.2
    and the directed graph G = (V, E) formalised in §6.4:

        V = {gate_id strings}   — checkpoints with camera coverage
        E = {(u, v, w)}         — directed edges weighted by minimum
                                  travel time w in seconds

    Deliberately built on a pure-Python adjacency dictionary rather than
    importing networkx, to demonstrate explicit command of graph data
    structures during the viva.

    Core operations used by SpatialVerifier:
      get_min_travel_time(u, v) → edge weight or None  [O(1) dict lookup]
      has_any_path(u, v)        → BFS reachability     [O(V + E)]
    """

    def __init__(self) -> None:
        # Adjacency representation:
        # { source_gate_id: { dest_gate_id: min_travel_seconds } }
        # Two levels of dict give O(1) edge-weight lookups.
        self._adj: Dict[str, Dict[str, float]] = {}
        self._nodes: Set[str] = set()

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_node(self, node_id: str) -> None:
        """Register a gate checkpoint as a vertex in the graph."""
        self._nodes.add(node_id)
        if node_id not in self._adj:
            self._adj[node_id] = {}

    def add_edge(self, source: str, dest: str, min_seconds: float) -> None:
        """
        Add a directed edge source → dest with weight min_seconds.

        Implicitly adds both endpoints as nodes if not already present.
        Edges are directed: add_edge('A', 'B') does NOT imply A ← B.
        For bidirectional roads, call add_edge twice.
        """
        self.add_node(source)
        self.add_node(dest)
        self._adj[source][dest] = float(min_seconds)

    # ── Query ────────────────────────────────────────────────────────────────

    def get_min_travel_time(self, source: str, dest: str) -> Optional[float]:
        """
        Return the minimum required travel time (seconds) for edge source → dest.

        Returns None if no direct edge exists, which signals a PATH_VIOLATION
        to the SpatialVerifier.

        Interim Report §6.4: "Each edge is weighted with the minimum travel
        time in seconds, calculated from the road distance and the facility's
        internal speed limit."
        """
        return self._adj.get(source, {}).get(dest, None)

    def has_direct_edge(self, source: str, dest: str) -> bool:
        """True if a direct edge source → dest exists in the graph."""
        return dest in self._adj.get(source, {})

    def has_any_path(self, source: str, dest: str) -> bool:
        """
        BFS reachability check: is there ANY path from source to dest?

        Used to distinguish two categories of PATH_VIOLATION:
          - No path at all     → topological impossibility
          - Path exists via B  → checkpoint B was skipped (credential sharing)

        Time complexity: O(V + E) — standard BFS over the adjacency list.
        """
        if source == dest:
            return True
        if source not in self._nodes:
            return False

        visited: Set[str] = set()
        queue: List[str] = [source]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for neighbour in self._adj.get(current, {}):
                if neighbour == dest:
                    return True
                queue.append(neighbour)

        return False

    # ── Introspection ────────────────────────────────────────────────────────

    def get_nodes(self) -> Set[str]:
        """Return a copy of the vertex set."""
        return set(self._nodes)

    def get_edges(self) -> List[Tuple[str, str, float]]:
        """Return all edges as (source, dest, min_seconds) triples."""
        return [
            (src, dst, w)
            for src, neighbours in self._adj.items()
            for dst, w in neighbours.items()
        ]

    def __repr__(self) -> str:
        edges = self.get_edges()
        return f"FacilityGraph(nodes={self._nodes}, edges={edges})"

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_travel_times(cls, travel_times: dict) -> 'FacilityGraph':
        """
        Construct a FacilityGraph from the travel_times config dict used
        throughout the codebase:
            { ('GATE1', 'GATE2'): 120, ('GATE1', 'GATE3'): 300, ... }

        This factory allows SpatialVerifier to accept the same CONFIG
        dict format as before while internally using the proper graph model.
        """
        graph = cls()
        for (source, dest), min_seconds in travel_times.items():
            graph.add_edge(source, dest, min_seconds)
        return graph


# ─────────────────────────────────────────────────────────────────────────────
# SpatialVerifier — Core Fraud Detection Engine
# Interim Report §5.2 Class Diagram: SpatialTemporalVerifier
#   -vehicle_state: Dictionary
#   -travel_times: Dictionary
#   +check_entry(plate_number, gate_id): Tuple[Boolean, String]
#   -_log_fraud(...)
# ─────────────────────────────────────────────────────────────────────────────

# Default TTL for vehicle state records.
# A vehicle not seen at any gate for longer than this is evicted from memory.
# EDGE-001 fix: without TTL, vehicle_state grows forever
# (200 vehicles/day × 365 days = 73,000 entries/year).
DEFAULT_STATE_TTL_SECONDS: int = 3_600   # 1 hour


class SpatialVerifier:
    """
    Layer 3 — Spatial-Temporal Correlation (STC) Engine.

    Implements the core fraud detection algorithm from Interim Report §6.4:

        "The dockyard is modelled as a directed graph G = (V, E). Each time
        a vehicle is detected at a gate, the engine retrieves the vehicle's
        previous gate and the elapsed time since that detection. If the
        elapsed time is less than the edge weight for that route, the system
        raises a fraud alert. Elapsed times below 0.5 seconds trigger a
        Clone Attack classification, since no physical movement could account
        for such an interval; longer violations are classified as Speeding
        Violations, which may indicate credential sharing."

    Sprint 2 additions beyond the core algorithm:
      - FacilityGraph replaces raw dict as the graph model (§5.2 fulfilment).
      - FraudAlert dataclass replaces anonymous dict records (§5.2 fulfilment).
      - TTL-based eviction of stale vehicle_state entries (EDGE-001 fix).
      - Negative ΔT guard for NTP/clock anomalies (EDGE-002 fix).
    """

    def __init__(
        self,
        travel_times: dict,
        state_ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS,
    ) -> None:
        # Build the directed graph from the travel times config.
        # Fulfils FacilityGraph class from Interim Report §5.2.
        self.graph: FacilityGraph = FacilityGraph.from_travel_times(travel_times)

        # Per-vehicle location state.
        # Implements the VehicleState machine referenced in Interim Report §5.2.
        # { plate_number: {'gate': str, 'timestamp': datetime} }
        self.vehicle_state: Dict[str, dict] = {}

        # Ordered list of FraudAlert instances — the in-memory fraud log.
        # Each item is also persisted to SQLite via DatabaseManager.log_fraud_event().
        self.fraud_alerts: List[FraudAlert] = []

        # EDGE-001 FIX: TTL for state records.
        self.state_ttl_seconds: int = state_ttl_seconds

        # Diagnostic counters for viva dashboard.
        self._eviction_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def check_entry(self, plate_number: str, gate_id: str) -> Tuple[bool, str]:
        """
        Core STC Engine method — Interim Report §6.4, Algorithm 2.

        Evaluates whether a new detection event (plate_number at gate_id)
        is physically consistent with the vehicle's known history.

        Steps:
          1. Lazy eviction of stale vehicle_state records (EDGE-001).
          2. First-appearance → store state, return valid.
          3. Same-gate re-detection → update timestamp, return valid.
          4. Transition: query FacilityGraph for min required time.
             4a. No direct edge → PATH_VIOLATION.
             4b. ΔT ≥ min_required → valid transition.
             4c. ΔT < 0.5s → CLONE_ATTACK (CRITICAL).
             4d. ΔT < min_required → SPEEDING_VIOLATION (HIGH).

        Returns:
            (is_allowed: bool, reason: str)

        When fraud is detected, the corresponding FraudAlert is appended to
        self.fraud_alerts. The caller (main_system.py) should retrieve it via
        get_latest_fraud_alert() and persist it via DatabaseManager.log_fraud_event().
        """
        now = datetime.now()

        # ── EDGE-001 FIX: Lazy TTL eviction ──────────────────────────────────
        # Runs on every call. O(n) scan but n is bounded by
        # (facility vehicle count × TTL window), not by total system uptime.
        self._evict_stale_states(now)

        # ── Case 1: First system appearance ──────────────────────────────────
        if plate_number not in self.vehicle_state:
            self.vehicle_state[plate_number] = {
                'gate':      gate_id,
                'timestamp': now,
            }
            return (True, "System Entry")

        prev      = self.vehicle_state[plate_number]
        prev_gate = prev['gate']
        prev_time = prev['timestamp']

        # ── Case 2: Same gate re-detection ───────────────────────────────────
        # Update timestamp so ΔT is measured from the most recent appearance
        # at THIS gate, not from the vehicle's first entry into the system.
        if prev_gate == gate_id:
            self.vehicle_state[plate_number]['timestamp'] = now
            return (True, "Same Gate Update")

        # ── Case 3: Transition between different gates ────────────────────────

        # EDGE-002 FIX: Guard against negative ΔT.
        # Caused by: NTP clock sync stepping backward, manual time change,
        # or DST transition. Without this guard, actual_time < 0 would
        # always satisfy actual_time < min_required, firing a false
        # CLONE_ATTACK alert for a legitimate vehicle.
        actual_time = (now - prev_time).total_seconds()
        if actual_time < 0:
            print(
                f"⚠️  Clock anomaly for {plate_number}: "
                f"ΔT = {actual_time:.3f}s. Resetting state, no alert raised."
            )
            self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now}
            return (True, f"Clock Anomaly — State Reset (ΔT={actual_time:.3f}s)")

        # Query the FacilityGraph for the minimum required travel time.
        # This is the edge weight lookup in G = (V, E) — Interim Report §6.4.
        min_required = self.graph.get_min_travel_time(prev_gate, gate_id)

        # ── Case 3a: No direct edge in graph → PATH_VIOLATION ─────────────────
        if min_required is None:
            # BFS to distinguish "no route" from "checkpoint skipped".
            path_exists = self.graph.has_any_path(prev_gate, gate_id)
            if path_exists:
                detail = "skipped intermediate checkpoint"
            else:
                detail = "no route defined in facility graph"

            reason = f"PATH VIOLATION: {prev_gate} → {gate_id} ({detail})"

            alert = self._build_fraud_alert(
                plate_number  = plate_number,
                origin_gate   = prev_gate,
                dest_gate     = gate_id,
                violation_type= 'PATH_VIOLATION',
                severity_level= 'HIGH',
                min_time      = 0.0,
                delta_time    = actual_time,
                reason        = reason,
            )
            self._log_fraud(alert)
            self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now}
            return (False, reason)

        # ── Case 3b: Valid transition — physics satisfied ─────────────────────
        if actual_time >= min_required:
            self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now}
            return (True, f"Valid: {actual_time:.1f}s (min {min_required:.0f}s)")

        # ── Case 3c / 3d: Fraud — ΔT < minimum required time ─────────────────
        #
        # Classification threshold from Interim Report §6.4:
        # < 0.5s  → CLONE_ATTACK  (CRITICAL severity)
        #   else  → SPEEDING_VIOLATION (HIGH severity)
        if actual_time < 0.5:
            violation_type = 'CLONE_ATTACK'
            severity_level = 'CRITICAL'
            reason = (
                f"CLONE ATTACK DETECTED: {prev_gate} → {gate_id} "
                f"in {actual_time:.3f}s (minimum {min_required:.0f}s) — "
                f"physically impossible; duplicate plate suspected"
            )
        else:
            violation_type = 'SPEEDING_VIOLATION'
            severity_level = 'HIGH'
            reason = (
                f"PHYSICS VIOLATION: {prev_gate} → {gate_id} "
                f"in {actual_time:.1f}s (minimum {min_required:.0f}s) — "
                f"exceeded speed limit; credential sharing suspected"
            )

        alert = self._build_fraud_alert(
            plate_number   = plate_number,
            origin_gate    = prev_gate,
            dest_gate      = gate_id,
            violation_type = violation_type,
            severity_level = severity_level,
            min_time       = min_required,
            delta_time     = actual_time,
            reason         = reason,
        )
        self._log_fraud(alert)
        self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now}
        return (False, reason)

    def get_fraud_report(self) -> List[dict]:
        """Return all FraudAlert records serialised as dicts (JSON-safe)."""
        return [alert.to_dict() for alert in self.fraud_alerts]

    def get_latest_fraud_alert(self) -> Optional[FraudAlert]:
        """
        Return the most recently logged FraudAlert instance.
        Called by main_system.py to pass the alert to DatabaseManager.log_fraud_event().
        """
        return self.fraud_alerts[-1] if self.fraud_alerts else None

    def get_stats(self) -> dict:
        """Diagnostic summary — useful for the viva live dashboard."""
        return {
            'tracked_vehicles':   len(self.vehicle_state),
            'total_fraud_alerts': len(self.fraud_alerts),
            'evicted_records':    self._eviction_count,
            'graph_nodes':        len(self.graph.get_nodes()),
            'graph_edges':        len(self.graph.get_edges()),
            'state_ttl_seconds':  self.state_ttl_seconds,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_fraud_alert(
        self,
        plate_number:   str,
        origin_gate:    str,
        dest_gate:      str,
        violation_type: str,
        severity_level: str,
        min_time:       float,
        delta_time:     float,
        reason:         str,
    ) -> FraudAlert:
        """
        Instantiate a FraudAlert dataclass.

        Fulfils the FraudAlert class from the Interim Report §5.2 class diagram
        which specifies: +plate_number, +violation_type, +severity_level,
        +delta_time as mandatory fields.
        """
        return FraudAlert(
            timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            plate_number   = plate_number,
            origin_gate    = origin_gate,
            dest_gate      = dest_gate,
            violation_type = violation_type,
            severity_level = severity_level,
            min_time       = min_time,
            delta_time     = round(delta_time, 3),
            reason         = reason,
        )

    def _log_fraud(self, alert: FraudAlert) -> None:
        """Append FraudAlert to the in-memory list and print to console."""
        self.fraud_alerts.append(alert)
        print(
            f"\n🚨  [SECURITY ALERT] [{alert.severity_level}] "
            f"{alert.plate_number}: {alert.reason}"
        )

    def _evict_stale_states(self, now: datetime) -> None:
        """
        EDGE-001 FIX: Remove vehicle_state entries older than state_ttl_seconds.

        Without eviction:
          200 vehicles/day × 365 days = 73,000 dict entries/year → memory leak.

        With 1-hour TTL:
          Dict is bounded to vehicles active in the last hour — the operationally
          meaningful window for any single shift period.

        Strategy: lazy eviction (runs on every check_entry call).
        Cost: O(n) per call, but n is bounded by facility size × TTL,
        not by system uptime.
        """
        cutoff = self.state_ttl_seconds
        stale = [
            plate
            for plate, state in self.vehicle_state.items()
            if (now - state['timestamp']).total_seconds() > cutoff
        ]
        for key in stale:
            del self.vehicle_state[key]
        if stale:
            self._eviction_count += len(stale)
