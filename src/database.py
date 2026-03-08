"""
src/database.py
Local SQLite Persistence Layer — POC Standalone Version

Sprint 5 Change (POC Decoupling):
  The Colombo Dockyard REST API (CDL backend) and its supporting infrastructure
  have been completely removed. The system is now 100% offline and standalone:

    REMOVED:
      - import requests (external HTTP library)
      - import queue    (async worker queue)
      - __init__ params: base_url, cam_code, device, in_out
      - insert_endpoint / select_endpoint URL construction
      - insert_queue (bounded async Queue, maxsize=500)
      - _process_insert_queue worker thread
      - _perform_api_insert() method (SEC-001 guard, BUG-004, BUG-005 all lived here)
      - shutdown() queue drain / thread join logic
      - sync parameter from insert_plate_detection()

    RATIONALE:
      Relying on an external network endpoint (esystems.cdl.lk) during a Viva
      defence is an unacceptable single point of failure. A lost WiFi connection
      or a dockyard server outage would crash the demonstration at the worst
      possible moment. The POC is now self-contained: every detection and fraud
      event is persisted to a local audit_log.db file only.

      The SEC-001 plate format validation guard that previously lived in
      _perform_api_insert() has been moved to insert_plate_detection() so
      that malformed OCR strings are still rejected before any DB write.
      The pipeline-level guard in main_system.py (_SL_PLATE_RE) remains
      unchanged as the first line of defence.

    RETAINED (unchanged):
      - SQLite dual-table schema (access_log, fraud_events, registered_vehicles)
      - WAL journal mode for concurrent-write semantics
      - Threading lock on all write paths
      - Full CRUD API for registered_vehicles (dashboard.py)
      - log_fraud_event(), insert_plate_detection()
      - query_recent_access_log(), query_recent_fraud_events()
      - seed_registered_vehicles(), get_registered_plates(), get_all_vehicles()

Database file: audit_log.db (path configurable at construction time)
Schema:
  access_log          — every plate detection event (Interim Report §5.3 ER)
  fraud_events        — every STC engine fraud alert
  registered_vehicles — live vehicle registry (managed via dashboard CRUD)
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.spatial import FraudAlert

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SQLITE_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'audit_log.db'
)

# SEC-001: canonical Sri Lankan plate format regex.
# Rejects malformed OCR output before any SQLite write.
# Accepts 2-letter ('WP-1234') and 3-letter ('CAB-1234') formats.
_SL_PLATE_RE = re.compile(r'^[A-Z]{2,3}-\d{4}$')

# ─────────────────────────────────────────────────────────────────────────────
# SQLite schema
# ─────────────────────────────────────────────────────────────────────────────

_SQLITE_SCHEMA: str = """
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS access_log (
        log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        plate_number    TEXT    NOT NULL,
        gate_id         TEXT    NOT NULL DEFAULT '',
        ocr_confidence  REAL    NOT NULL DEFAULT 0.0,
        decision        TEXT    NOT NULL DEFAULT 'UNKNOWN',
        override_status TEXT    NOT NULL DEFAULT 'NONE'
    );
    CREATE INDEX IF NOT EXISTS idx_access_plate
        ON access_log (plate_number);
    CREATE INDEX IF NOT EXISTS idx_access_time
        ON access_log (timestamp);

    CREATE TABLE IF NOT EXISTS fraud_events (
        event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        plate_number    TEXT    NOT NULL,
        violation_type  TEXT    NOT NULL,
        severity_level  TEXT    NOT NULL,
        origin_gate     TEXT    NOT NULL DEFAULT '',
        dest_gate       TEXT    NOT NULL DEFAULT '',
        delta_time      REAL,
        reason          TEXT    NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_fraud_plate
        ON fraud_events (plate_number);
    CREATE INDEX IF NOT EXISTS idx_fraud_type
        ON fraud_events (violation_type);

    CREATE TABLE IF NOT EXISTS registered_vehicles (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number    TEXT    NOT NULL UNIQUE,
        owner_name      TEXT    NOT NULL DEFAULT '',
        vehicle_type    TEXT    NOT NULL DEFAULT 'Car',
        department      TEXT    NOT NULL DEFAULT '',
        registered_date TEXT    NOT NULL,
        is_active       INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX IF NOT EXISTS idx_vehicles_plate
        ON registered_vehicles (plate_number);
"""

# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Local SQLite persistence layer for the VLPR POC system.

    All writes go directly to audit_log.db — no network calls, no async queue,
    no external dependencies. Designed for reliable offline Viva demonstration.

    Thread safety:
      A single threading.Lock() serialises all SQLite write operations.
      Read operations (queries) open short-lived connections without the lock
      since SQLite WAL mode supports concurrent readers.

    Usage:
        db = DatabaseManager(sqlite_db_path='audit_log.db')
        db.insert_plate_detection('CAB-1234', 0.94, 1, gate_id='GATE_A')
        db.log_fraud_event(fraud_alert)
        db.shutdown()   # no-op in POC mode; kept for API compatibility
    """

    def __init__(
        self,
        sqlite_db_path: str = DEFAULT_SQLITE_PATH,
    ) -> None:
        """
        Initialise the local SQLite database.

        Args:
            sqlite_db_path: Path to the audit_log.db file. Created if absent.
        """
        self.sqlite_db_path = os.path.abspath(sqlite_db_path)
        self._sqlite_lock   = threading.Lock()

        # Stats counters — kept for dashboard /api/health compatibility.
        self.stats: dict = {
            'total_inserts':      0,
            'successful_inserts': 0,
            'failed_inserts':     0,
        }

        self._init_sqlite()
        print(f"    SQLite audit log → {self.sqlite_db_path}")

    # ── Schema initialisation ─────────────────────────────────────────────────

    def _init_sqlite(self) -> None:
        """Create tables and indices if they do not already exist."""
        try:
            os.makedirs(os.path.dirname(self.sqlite_db_path), exist_ok=True)
        except FileNotFoundError:
            pass  # path is in the current directory — no makedirs needed

        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.executescript(_SQLITE_SCHEMA)
                    conn.commit()
            print(f"    SQLite schema ready ({self.sqlite_db_path})")
        except sqlite3.Error as e:
            print(f"⚠️  SQLite init error: {e}")

    # ── Core write: access_log ────────────────────────────────────────────────

    def insert_plate_detection(
        self,
        plate_number:    str,
        ocr_confidence:  float,
        track_id:        int,
        gate_id:         str   = '',
        decision:        str   = 'GRANTED',
        override_status: str   = 'NONE',
    ) -> bool:
        """
        Write a plate detection event to the local access_log table.

        Sprint 5: all REST API logic removed. This method now performs one
        operation: a single SQLite INSERT. It is synchronous, lock-protected,
        and requires no network connectivity.

        SEC-001 guard retained: plate strings that fail the canonical SL
        format regex are rejected before any write. The pipeline-level guard
        in main_system.py catches most bad strings earlier; this is the
        final backstop inside the persistence layer.

        Args:
            plate_number:    Canonical plate string (e.g. 'CAB-1234').
            ocr_confidence:  Average character recognition confidence [0-1].
            track_id:        Internal tracker ID (for cross-referencing logs).
            gate_id:         Gate identifier string (e.g. 'GATE_A').
            decision:        'GRANTED' | 'DENIED' | 'FRAUD'.
            override_status: Manual override flag (default 'NONE').

        Returns:
            True on successful INSERT, False if format validation fails or
            a SQLite error occurs.
        """
        # ── SEC-001: format guard ─────────────────────────────────────────────
        if not _SL_PLATE_RE.match(plate_number):
            print(
                f"🔒 SEC-001: '{plate_number}' failed format validation "
                f"— SQLite write suppressed."
            )
            self.stats['failed_inserts'] += 1
            return False
        # ── End SEC-001 ───────────────────────────────────────────────────────

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.execute(
                        """INSERT INTO access_log
                           (timestamp, plate_number, gate_id,
                            ocr_confidence, decision, override_status)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (timestamp, plate_number, gate_id,
                         round(ocr_confidence, 4), decision, override_status)
                    )
                    conn.commit()
            self.stats['total_inserts']      += 1
            self.stats['successful_inserts'] += 1
            print(f"🗄️  Logged: {plate_number} | {gate_id} | {decision} | conf={ocr_confidence:.2f}")
            return True

        except sqlite3.Error as e:
            self.stats['total_inserts']  += 1
            self.stats['failed_inserts'] += 1
            print(f"⚠️  SQLite insert error for '{plate_number}': {e}")
            return False

    # ── Core write: fraud_events ──────────────────────────────────────────────

    def log_fraud_event(self, alert: 'FraudAlert') -> bool:
        """
        Persist a FraudAlert dataclass instance to the fraud_events table.

        Called by main_system.py and research_demo.py immediately after
        SpatialVerifier.check_entry() returns (False, reason).

        Args:
            alert: A FraudAlert dataclass instance from src/spatial.py.

        Returns:
            True on successful INSERT, False on SQLite error.
        """
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.execute(
                        """INSERT INTO fraud_events
                           (timestamp, plate_number, violation_type,
                            severity_level, origin_gate, dest_gate,
                            delta_time, reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            alert.timestamp,
                            alert.plate_number,
                            alert.violation_type,
                            alert.severity_level,
                            alert.origin_gate,
                            alert.dest_gate,
                            round(alert.delta_time, 4),
                            alert.reason,
                        )
                    )
                    conn.commit()
            print(f"🗄️  Fraud event logged: [{alert.violation_type}] {alert.plate_number}")
            return True
        except sqlite3.Error as e:
            print(f"⚠️  SQLite fraud log error: {e}")
            return False

    # ── Registered Vehicle Registry — Sprint 4 (unchanged) ───────────────────

    def seed_registered_vehicles(self) -> None:
        """
        Populate the registered_vehicles table from the hardcoded
        REGISTERED_VEHICLES frozenset if the table is currently empty.

        Idempotent — subsequent calls are no-ops.
        """
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    count = conn.execute(
                        'SELECT COUNT(*) FROM registered_vehicles'
                    ).fetchone()[0]
                    if count > 0:
                        return

                    from src.validator import REGISTERED_VEHICLES
                    today = datetime.now().strftime('%Y-%m-%d')

                    dept_map = {
                        'WP': 'Operations',  'SP': 'Logistics',
                        'NW': 'Engineering', 'EP': 'Security',
                        'NC': 'Finance',     'SG': 'Administration',
                        'CAB': 'Management', 'UVA': 'Maintenance',
                        'SAB': 'IT',         'NCP': 'HR',
                    }

                    rows = []
                    for plate in sorted(REGISTERED_VEHICLES):
                        prefix = plate.split('-')[0]
                        dept   = dept_map.get(prefix, 'General')
                        rows.append((plate, f'Employee ({prefix})', 'Car', dept, today, 1))

                    conn.executemany(
                        """INSERT OR IGNORE INTO registered_vehicles
                           (plate_number, owner_name, vehicle_type,
                            department, registered_date, is_active)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                    conn.commit()
                    print(f"    Seeded {len(rows)} vehicles into registered_vehicles table.")
        except sqlite3.Error as e:
            print(f"⚠️  Vehicle seed error: {e}")

    def get_registered_plates(self) -> list:
        """Return list of active plate strings. Used by CV pipeline (Sprint 4)."""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                rows = conn.execute(
                    "SELECT plate_number FROM registered_vehicles WHERE is_active=1"
                ).fetchall()
                return [r[0] for r in rows]
        except sqlite3.Error as e:
            print(f"⚠️  get_registered_plates error: {e}")
            return []

    def get_all_vehicles(self) -> list:
        """Return all registered_vehicles rows as list of dicts (dashboard)."""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT id, plate_number, owner_name, vehicle_type,
                              department, registered_date, is_active
                       FROM registered_vehicles
                       ORDER BY plate_number"""
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            print(f"⚠️  get_all_vehicles error: {e}")
            return []

    def add_vehicle(
        self,
        plate_number: str,
        owner_name:   str,
        vehicle_type: str,
        department:   str,
    ) -> tuple:
        """INSERT a new vehicle. Returns (success: bool, message: str)."""
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.execute(
                        """INSERT INTO registered_vehicles
                           (plate_number, owner_name, vehicle_type,
                            department, registered_date, is_active)
                           VALUES (?, ?, ?, ?, ?, 1)""",
                        (
                            plate_number.strip().upper(),
                            owner_name.strip(),
                            vehicle_type.strip(),
                            department.strip(),
                            datetime.now().strftime('%Y-%m-%d'),
                        )
                    )
                    conn.commit()
                    return (True, f"Vehicle {plate_number} registered successfully.")
        except sqlite3.IntegrityError:
            return (False, f"Plate {plate_number} already exists in the registry.")
        except sqlite3.Error as e:
            return (False, f"Database error: {e}")

    def update_vehicle(
        self,
        vehicle_id:   int,
        owner_name:   str,
        vehicle_type: str,
        department:   str,
        is_active:    int,
    ) -> tuple:
        """UPDATE a vehicle record. Returns (success, message)."""
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.execute(
                        """UPDATE registered_vehicles
                           SET owner_name=?, vehicle_type=?, department=?, is_active=?
                           WHERE id=?""",
                        (owner_name, vehicle_type, department, int(is_active), vehicle_id)
                    )
                    conn.commit()
                    return (True, "Vehicle updated successfully.")
        except sqlite3.Error as e:
            return (False, f"Database error: {e}")

    def delete_vehicle(self, vehicle_id: int) -> tuple:
        """Hard-delete a vehicle record. Returns (success, message)."""
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.execute(
                        "DELETE FROM registered_vehicles WHERE id=?", (vehicle_id,)
                    )
                    conn.commit()
                    return (True, "Vehicle removed from registry.")
        except sqlite3.Error as e:
            return (False, f"Database error: {e}")

    # ── Dashboard query helpers ───────────────────────────────────────────────

    def query_recent_access_log(self, limit: int = 20) -> list:
        """
        Retrieve the most recent access log entries for dashboard display.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts with keys: log_id, timestamp, plate_number,
            gate_id, ocr_confidence, decision.
        """
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT log_id, timestamp, plate_number, gate_id,
                              ocr_confidence, decision
                       FROM access_log
                       ORDER BY log_id DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            print(f"⚠️  query_recent_access_log error: {e}")
            return []

    def query_recent_fraud_events(self, limit: int = 20) -> list:
        """
        Retrieve the most recent fraud events for the dashboard fraud feed.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts with keys matching the fraud_events schema.
        """
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT event_id, timestamp, plate_number,
                              violation_type, severity_level,
                              origin_gate, dest_gate, delta_time, reason
                       FROM fraud_events
                       ORDER BY event_id DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            print(f"⚠️  query_recent_fraud_events error: {e}")
            return []

    # ── Stats & lifecycle ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return operational statistics for the dashboard /api/health endpoint.

        Returns:
            Dict with insert counters and SQLite row counts.
        """
        stats = dict(self.stats)
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                stats['access_log_rows']   = conn.execute(
                    'SELECT COUNT(*) FROM access_log').fetchone()[0]
                stats['fraud_event_rows']  = conn.execute(
                    'SELECT COUNT(*) FROM fraud_events').fetchone()[0]
                stats['registered_vehicles'] = conn.execute(
                    'SELECT COUNT(*) FROM registered_vehicles').fetchone()[0]
        except sqlite3.Error:
            pass
        return stats

    def shutdown(self) -> None:
        """
        Graceful shutdown.

        Sprint 5: no-op in POC mode — there is no queue to drain and no
        background thread to stop. Kept in the public API so all callers
        (main_system.py, research_demo.py, dashboard.py) remain unchanged.
        """
        print("🗄️  DatabaseManager shut down (local SQLite, no queue to drain).")
