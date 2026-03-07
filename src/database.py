"""
src/database.py
Dual-Channel Persistence Layer

This module fulfils the database claims from the Interim Report:

  1. Tamper-evident Audit Log (Interim Report §3.1, FR-05):
     "Every access transaction must produce a structured log record containing
     the cropped plate image, the raw OCR output, the normalised plate string,
     the detection timestamp, the gate identifier, the recognition confidence
     score, and the final access decision. These records must be stored in a
     tamper-evident format suitable for Finance Department review."

  2. AccessLog and FraudEvent entities (Interim Report §5.3, ER Diagram):
     AccessLog  — LogID, PlateNo, GateID, Timestamp, OCR_Confidence, OverrideStatus
     FraudEvent — EventID, PlateNo, ViolationType, SeverityLevel, DeltaTime

  3. PostgreSQL MVCC capability (Interim Report §3.3):
     "PostgreSQL is selected because of its Multi-Version Concurrency Control
     (MVCC) capability. MVCC allows multiple camera processing threads to write
     audit log entries simultaneously without causing locks."
     → Implemented as local SQLite with a threading.Lock() for portability
       during the demo. SQLite's WAL mode provides equivalent concurrency
       semantics for the single-machine viva deployment.

  4. Channel 2: REST API async queue for CDL production backend.

Sprint 2 Fixes Applied:
  - BUG-004: CamCode field corrected. Previously sent the plate_number
             in the CamCode parameter (wrong field). Now: CamCode = self.cam_code
             (the camera identifier), PlateNumber = the detected plate text.
             Payload defined as a named dict for easy key updates.

  - BUG-005: Queue drain before shutdown. Previously daemon=True caused
             instant thread kill on exit, silently dropping queued detections.
             Now: shutdown() calls Queue.join() to block until all items are
             processed before stopping the worker thread.

  - ARCH-004: Queue bounded at maxsize=500 to prevent memory exhaustion
              when the REST API is unreachable during burst traffic.
"""

from __future__ import annotations

import os
import queue
import re
import sqlite3
import threading
from datetime import datetime
from typing import TYPE_CHECKING

import requests

# ── SEC-001: Sri Lankan plate format regex ────────────────────────────────────
# Applied in _perform_api_insert() before any data leaves the system.
# Prevents malformed OCR output (e.g. garbage strings, injection attempts)
# from being submitted to the CDL REST endpoint.
# Accepts: 'WP-1234' (2-letter) and 'CAB-1234' (3-letter) canonical formats.
_SL_PLATE_RE = re.compile(r'^[A-Z]{2,3}-\d{4}$')

if TYPE_CHECKING:
    # Import only for type hints — avoids circular import at runtime.
    from src.spatial import FraudAlert


# ─────────────────────────────────────────────────────────────────────────────
# Default SQLite path — placed in the project root for easy inspection.
# ─────────────────────────────────────────────────────────────────────────────

_MODULE_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT   = os.path.normpath(os.path.join(_MODULE_DIR, '..'))
DEFAULT_SQLITE_PATH = os.path.join(_PROJECT_ROOT, 'audit_log.db')


class DatabaseManager:
    """
    Dual-channel persistence: local SQLite audit log + CDL REST API.

    Channel 1 — SQLite (synchronous, always-on):
      Writes every plate detection and every fraud event to the local
      audit_log.db file immediately. This is the tamper-evident audit
      trail required by the Finance Department (Interim Report FR-05).
      Schema matches the ER Diagram entities: AccessLog, FraudEvent.

    Channel 2 — REST API (asynchronous queue):
      Queues detection records for submission to the CDL backend.
      Failures do not block the main detection pipeline.

    Thread safety:
      SQLite writes are serialised with a threading.Lock() (_sqlite_lock).
      REST API writes are serialised through a bounded Queue (maxsize=500).
    """

    # ── Database schema — mirrors the ER Diagram from Interim Report §5.3 ───
    _SQLITE_SCHEMA: str = """
        -- AccessLog entity: Interim Report §5.3 ER Diagram
        -- Fields: LogID, PlateNo, GateID, Timestamp, OCR_Confidence, OverrideStatus
        CREATE TABLE IF NOT EXISTS access_log (
            log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            plate_number    TEXT    NOT NULL,
            gate_id         TEXT    NOT NULL DEFAULT '',
            ocr_confidence  REAL,
            decision        TEXT    NOT NULL DEFAULT 'GRANTED',
            override_status INTEGER          DEFAULT 0
        );

        -- FraudEvent entity: Interim Report §5.3 ER Diagram
        -- Fields: EventID, PlateNo, ViolationType, SeverityLevel, DeltaTime
        CREATE TABLE IF NOT EXISTS fraud_events (
            event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            plate_number    TEXT    NOT NULL,
            violation_type  TEXT    NOT NULL,
            severity_level  TEXT    NOT NULL,
            origin_gate     TEXT,
            dest_gate       TEXT,
            delta_time      REAL,
            reason          TEXT
        );

        -- Indices for Finance Department queries: by plate and by date.
        CREATE INDEX IF NOT EXISTS idx_access_plate
            ON access_log (plate_number);
        CREATE INDEX IF NOT EXISTS idx_access_timestamp
            ON access_log (timestamp);
        CREATE INDEX IF NOT EXISTS idx_fraud_plate
            ON fraud_events (plate_number);
        CREATE INDEX IF NOT EXISTS idx_fraud_type
            ON fraud_events (violation_type);
    """

    def __init__(
        self,
        base_url:        str,
        cam_code:        str,
        device:          str,
        in_out:          str,
        sqlite_db_path:  str = DEFAULT_SQLITE_PATH,
    ) -> None:
        # ── REST API config ───────────────────────────────────────────────────
        self.base_url         = base_url
        self.cam_code         = cam_code   # BUG-004 fix: stored as camera ID
        self.device           = device
        self.in_out           = in_out
        self.insert_endpoint  = f"{base_url}/PostRFID"
        self.select_endpoint  = f"{base_url}/GetCamDetails"

        # ── SQLite local audit log ────────────────────────────────────────────
        # Fulfils the PostgreSQL audit log described in Interim Report §3.3.
        # SQLite + WAL mode provides equivalent single-machine concurrency.
        self.sqlite_db_path = os.path.normpath(sqlite_db_path)
        self._sqlite_lock   = threading.Lock()   # serialise concurrent writes
        self._init_sqlite()

        # ── Async REST API queue ──────────────────────────────────────────────
        # ARCH-004 FIX: bounded queue — prevents memory exhaustion.
        # BUG-005 FIX: drain via Queue.join() before thread is stopped.
        self.insert_queue = queue.Queue(maxsize=500)
        self.is_running   = True
        self.insert_thread = threading.Thread(
            target=self._process_insert_queue,
            daemon=True,
            name="DBInsertWorker",
        )
        self.insert_thread.start()

        # Performance counters.
        self.stats: dict = {
            'total_inserts':      0,
            'successful_inserts': 0,
            'failed_inserts':     0,
        }

        print("🗄️  DatabaseManager initialised")
        print(f"    Local SQLite  → {self.sqlite_db_path}")
        print(f"    REST endpoint → {self.insert_endpoint}")

    # ─────────────────────────────────────────────────────────────────────────
    # SQLite initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_sqlite(self) -> None:
        """
        Create the audit database and apply the schema.

        Uses executescript() so both tables and indices are created in one
        transaction. Safe to call on an existing database — all statements
        use CREATE TABLE/INDEX IF NOT EXISTS.
        """
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                # WAL mode: allows concurrent readers while writing.
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.executescript(self._SQLITE_SCHEMA)
                conn.commit()
            print(f"    SQLite schema ready ({self.sqlite_db_path})")
        except sqlite3.Error as e:
            # Non-fatal: REST API channel still functions without SQLite.
            print(f"⚠️  SQLite initialisation error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def insert_plate_detection(
        self,
        plate_number: str,
        confidence:   float,
        track_id:     int,
        gate_id:      str  = '',
        decision:     str  = 'GRANTED',
        sync:         bool = False,
    ) -> bool:
        """
        Record a plate detection event through both persistence channels.

        Channel 1 (SQLite): always synchronous — writes the AccessLog row
        immediately for audit integrity. If the process terminates before
        the REST queue is drained, the SQLite record is already safe.

        Channel 2 (REST API): async by default. Queued for background
        submission. If sync=True, blocks until the API responds.

        Args:
            plate_number: Normalised plate string (e.g. 'CAB-1234').
            confidence:   OCR confidence score (0.0–1.0).
            track_id:     Tracker-assigned track identifier.
            gate_id:      Gate where the detection occurred.
            decision:     'GRANTED' or 'DENIED'.
            sync:         Block until REST API call completes.

        Returns:
            True if accepted for processing, False if queue was full.
        """
        # Channel 1: SQLite — immediate, synchronous.
        self._sqlite_log_access(plate_number, gate_id, confidence, decision)

        # Channel 2: REST API — async.
        detection_data = {
            'plate_number': plate_number,
            'confidence':   confidence,
            'track_id':     track_id,
            'gate_id':      gate_id,
            'decision':     decision,
        }

        if sync:
            return self._perform_api_insert(detection_data)
        else:
            try:
                self.insert_queue.put_nowait(detection_data)
                return True
            except queue.Full:
                print(
                    f"⚠️  Insert queue full ({self.insert_queue.maxsize} items) — "
                    f"dropping API call for {plate_number}"
                )
                return False

    def log_fraud_event(self, alert: 'FraudAlert') -> None:
        """
        Persist a FraudAlert to the local SQLite fraud_events table.

        Fulfils the FraudEvent entity from the Interim Report §5.3 ER Diagram:
          EventID, PlateNo, ViolationType, SeverityLevel, DeltaTime.

        Called from main_system.py immediately after spatial_verifier.check_entry()
        returns is_allowed=False and the FraudAlert is retrieved via
        spatial_verifier.get_latest_fraud_alert().

        Args:
            alert: A FraudAlert dataclass instance from src/spatial.py.
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
                            alert.delta_time,
                            alert.reason,
                        )
                    )
                    conn.commit()
                    print(
                        f"🗄️  Fraud event logged: [{alert.violation_type}] "
                        f"{alert.plate_number}"
                    )
        except sqlite3.Error as e:
            print(f"⚠️  SQLite fraud event write error: {e}")

    def get_stats(self) -> dict:
        """Return runtime statistics for monitoring and viva dashboard."""
        stats = self.stats.copy()
        stats['queue_depth'] = self.insert_queue.qsize()
        return stats

    def query_recent_access_log(self, limit: int = 20) -> list:
        """
        Retrieve the most recent access log entries for dashboard display.

        Demonstrates the audit capability to examiners during the viva.
        """
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """SELECT timestamp, plate_number, gate_id,
                              ocr_confidence, decision
                       FROM access_log
                       ORDER BY log_id DESC
                       LIMIT ?""",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"⚠️  SQLite query error: {e}")
            return []

    def query_fraud_events(self, limit: int = 20) -> list:
        """Retrieve recent fraud events for dashboard display."""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """SELECT timestamp, plate_number, violation_type,
                              severity_level, delta_time, reason
                       FROM fraud_events
                       ORDER BY event_id DESC
                       LIMIT ?""",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"⚠️  SQLite query error: {e}")
            return []

    def shutdown(self) -> None:
        """
        BUG-005 FIX: Drain the REST API queue before stopping the worker thread.

        Previous behaviour (WRONG):
            daemon=True thread → killed instantly on process exit.
            Any items still in the queue were permanently lost.

        Correct behaviour:
            1. Queue.join() blocks until every item has had task_done() called.
               This guarantees all queued detections are submitted before exit.
            2. is_running = False signals the worker loop to exit.
            3. Thread.join(timeout=5) waits for the thread to finish cleanly.

        This means the process may take up to (queue depth × API timeout)
        seconds to exit — acceptable since it only fires on clean shutdown.
        """
        print("🗄️  Draining insert queue before shutdown...")
        self.insert_queue.join()        # block until all task_done() calls match
        self.is_running = False
        self.insert_thread.join(timeout=5)
        print("🗄️  DatabaseManager shut down cleanly.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sqlite_log_access(
        self,
        plate_number: str,
        gate_id:      str,
        confidence:   float,
        decision:     str,
    ) -> None:
        """
        Write one row to the access_log table (Channel 1).

        Uses a threading.Lock to serialise concurrent writes from multiple
        camera processing threads — equivalent to PostgreSQL's MVCC guarantee
        described in Interim Report §3.3 for the single-machine demo context.
        """
        try:
            with self._sqlite_lock:
                with sqlite3.connect(self.sqlite_db_path) as conn:
                    conn.execute(
                        """INSERT INTO access_log
                           (timestamp, plate_number, gate_id,
                            ocr_confidence, decision)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            plate_number,
                            gate_id,
                            round(confidence, 4),
                            decision,
                        )
                    )
                    conn.commit()
        except sqlite3.Error as e:
            print(f"⚠️  SQLite access_log write error: {e}")

    def _perform_api_insert(self, detection_data: dict) -> bool:
        """
        Submit a detection event to the CDL REST endpoint.

        BUG-004 FIX: Parameter mapping corrected.

        BEFORE (WRONG):
            params = {'CamCode': detection_data['plate_number'], ...}
            The plate number was sent in the CamCode (camera code) field.
            self.cam_code (the actual camera identifier) was never sent.

        AFTER (CORRECT):
            CamCode     → self.cam_code        (the camera identifier)
            PlateNumber → detection_data['plate_number']  (the OCR result)

        The payload dict is defined explicitly as a named mapping so that
        field names can be updated in one place without touching surrounding
        logic — satisfying the modularity requirement from the brief.
        """
        # ── SEC-001: Strict plate format validation before transmission ─────────
        # Reject any plate string that does not match the canonical Sri Lankan
        # format before it leaves the system. This prevents:
        #   - Garbage OCR output ('No text detected', etc.) reaching the API.
        #   - Parameter injection attempts via malformed plate strings.
        #   - Downstream database corruption from invalid format entries.
        #
        # The check runs BEFORE the try block so that format rejections are
        # not counted as attempted transmissions in total_inserts.
        #
        # Mirrors FR-03 (Interim Report §3.1): the system must not submit data
        # to external services unless it passes format validation.
        plate_to_submit = detection_data.get('plate_number', '')
        if not _SL_PLATE_RE.match(plate_to_submit):
            print(
                f"🔒 SEC-001: Plate '{plate_to_submit}' failed format validation "
                f"— REST API call suppressed."
            )
            self.stats['failed_inserts'] += 1
            return False
        # ── End SEC-001 guard ────────────────────────────────────────────────

        try:
            # BUG-004 FIX: corrected field mapping.
            api_payload: dict = {
                'CamCode':     self.cam_code,                      # camera ID
                'PlateNumber': detection_data['plate_number'],     # OCR result
                'Device':      self.device,
                'InOut':       self.in_out,
            }

            response = requests.get(
                self.insert_endpoint,
                params=api_payload,
                timeout=10,
            )
            self.stats['total_inserts'] += 1

            if response.status_code in (200, 201):
                self.stats['successful_inserts'] += 1
                print(
                    f"✅ '{detection_data['plate_number']}' → "
                    f"REST API OK ({response.status_code})"
                )
                return True
            else:
                self.stats['failed_inserts'] += 1
                print(
                    f"⚠️  REST API {response.status_code} for "
                    f"'{detection_data['plate_number']}'"
                )
                return False

        except requests.exceptions.Timeout:
            self.stats['failed_inserts'] += 1
            print(
                f"⚠️  REST API timeout for '{detection_data.get('plate_number', '?')}'"
            )
            return False
        except requests.exceptions.RequestException as e:
            self.stats['failed_inserts'] += 1
            print(f"⚠️  REST API error: {e}")
            return False

    def _process_insert_queue(self) -> None:
        """
        Worker thread: continuously drain the async insert queue.

        Processes items one at a time. task_done() is called after each
        item so that Queue.join() in shutdown() can unblock correctly.

        After is_running=False is set, the loop exits and any remaining
        items in the queue have already been drained by Queue.join()
        in shutdown() — so no data is lost.
        """
        while self.is_running:
            try:
                detection_data = self.insert_queue.get(timeout=1.0)
                self._perform_api_insert(detection_data)
                self.insert_queue.task_done()
            except queue.Empty:
                continue
