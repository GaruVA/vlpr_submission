# Multi-Point Spatial-Temporal Verification Framework
## for Fraud-Resistant Vehicle Access Control in RF-Congested Industrial Environments

**Student:** Garuka Assalaarachchi | **Index:** 10952592 | **BSc. Software Engineering**
**Supervisor:** Mr. Madusanka Mithrananda | **University of Plymouth**

---

## Project Overview

This system replaces the failing UHF RFID infrastructure at Colombo Dockyard PLC with a
physics-aware vehicle access control framework. Rather than authenticating a removable token,
it verifies the physical vehicle and checks whether its movement between gate checkpoints is
spatially and temporally possible.

The academic contribution is the **Layer 3 Spatial-Temporal Correlation (STC) Engine**, which
models the facility as a directed graph G = (V, E) and raises fraud alerts when a vehicle's
observed travel time between gates violates the physical minimum defined by road distance and
the facility speed limit.

---

## Three-Layer Architecture

| Layer | File | Academic Claim |
|---|---|---|
| **Layer 1** — Perception | `main_system.py`, `research_demo.py` | YOLOv8 plate localisation + CLAHE preprocessing (Interim Report §7.4, Snippet 2) |
| **Layer 2** — Correction | `src/validator.py` | LPM-MLED: Modified Levenshtein with optical confusion penalty matrix (Interim Report §6.4, Algorithm 1) |
| **Layer 3** — STC Engine | `src/spatial.py` | Directed graph fraud detection: CLONE_ATTACK, SPEEDING_VIOLATION, PATH_VIOLATION (Interim Report §6.4, Algorithm 2) |

---

## Repository Structure

```
vlpr_submission/
├── main_system.py          # Production pipeline (RTSP / full dockyard deployment)
├── research_demo.py        # Dual-camera tabletop Viva demo
├── requirements.txt        # Pinned dependencies
│
├── src/
│   ├── spatial.py          # FacilityGraph, FraudAlert, SpatialVerifier (Layer 3)
│   ├── validator.py        # SriLankanPlateValidator, LPM-MLED algorithm (Layer 2)
│   ├── tracker.py          # PlateTracker — consensual plate tracking (Layer 2)
│   ├── database.py         # SQLite audit log + CDL REST API (dual-channel persistence)
│   └── utils.py            # enhance_plate_contrast() CLAHE, smart_character_ordering()
│
└── models/                 # Place trained .pt files here (not included in submission)
    ├── plate_detection.pt      ← YOLOv8-Small, trained on 327 SL plate images
    └── character_recognition.pt ← YOLOv8 character classifier (37 classes)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** For CUDA-accelerated inference (RTX 3060, CUDA 11.8):
> ```bash
> pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 \
>             --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Add trained models

Place your `.pt` files in the `models/` directory:
```
models/plate_detection.pt
models/character_recognition.pt
```

### 3. Set the RTSP stream URL (production only)

```bash
export RTSP_URL="rtsp://user:password@192.168.x.x:554/Streaming/Channels/101"
```

Credentials are read from the environment variable. They are **not** stored in source code.

---

## Running the Viva Demo (Tabletop Dual-Camera)

Connect two webcams (indices 1 and 2). Run:

```bash
python research_demo.py
```

**Visual indicators:**
- 🟢 Green box — valid vehicle entry or legal gate transition
- 🔴 Red box + alert bar — fraud detected (CLONE_ATTACK or SPEEDING_VIOLATION)

**Physics constraint:** minimum 5 seconds travel time between Gate A and Gate B.
Present one toy vehicle at Gate A, then immediately at Gate B to trigger a CLONE_ATTACK alert.

---

## Running the Production System

```bash
python main_system.py
```

Or override the RTSP source at runtime:

```bash
python main_system.py "rtsp://user:pass@host:554/stream"
```

---

## Key Algorithm: LPM-MLED (src/validator.py)

The Modified Levenshtein Edit Distance assigns weighted substitution costs based on optical
character confusion:

| Pair | Cost | Rationale |
|---|---|---|
| `8` / `B` | 0.1 | Identical vertical bar structure under blur |
| `0` / `O` | 0.1 | Circular shapes indistinguishable at low resolution |
| `1` / `I` | 0.1 | Single vertical stroke |
| `5` / `S` | 0.1 | S-curve shapes |
| Any unrelated pair | 1.0 | Full standard penalty |

A match is accepted if `lpm_mled(ocr_output, registered_plate) ≤ 0.5`.

---

## Key Algorithm: FacilityGraph (src/spatial.py)

Pure-Python directed graph — no external graph library. The adjacency dictionary gives O(1)
edge-weight lookups. Fraud classification:

| Condition | Alert Type | Severity |
|---|---|---|
| ΔT < 0.5s | `CLONE_ATTACK` | CRITICAL |
| 0.5s ≤ ΔT < min_T | `SPEEDING_VIOLATION` | HIGH |
| No edge in graph | `PATH_VIOLATION` | HIGH |

---

## Audit Log

Every detection is written to `audit_log.db` (SQLite, WAL mode) before any network call.
Schema matches the Interim Report §5.3 ER Diagram:

- **access_log** — LogID, PlateNo, GateID, Timestamp, OCR_Confidence, Decision
- **fraud_events** — EventID, PlateNo, ViolationType, SeverityLevel, DeltaTime

Inspect the live audit log during the demo:
```bash
sqlite3 audit_log.db "SELECT * FROM access_log ORDER BY log_id DESC LIMIT 10;"
sqlite3 audit_log.db "SELECT * FROM fraud_events;"
```

---

## Test Coverage Summary

| Sprint | Tests | Result |
|---|---|---|
| Sprint 1 — Critical bug fixes | 12 | ✅ 12/12 |
| Sprint 2 — Academic algorithm implementation | 52 | ✅ 52/52 |
| Sprint 3 — Security, CLAHE, final polish | 60 | ✅ 60/60 |
