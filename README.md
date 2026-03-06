# Multi-Point Spatial-Temporal Verification Framework 
## for Fraud-Resistant Vehicle Access Control in RF-Congested Industrial Environments

### Project Overview
This repository contains the software implementation for the final year research project. The system pivots from simple ALPR to a physics-aware security framework that detects "impossible" vehicle movements (Cloning, Teleportation, and Path Violations) using Spatial-Temporal logic.

### Structure
- `src/`: Core logic modules.
  - `tracker.py`: Consensual Plate Tracking (Layer 2).
  - `spatial.py`: Spatial-Temporal Verification Engine (Layer 3).
  - `validator.py`: Sri Lankan Plate format validation.
  - `utils.py`: Helper functions.
  - `database.py`: Async database handler.
- `models/`: YOLOv8 models for plate detection and character recognition.
- `research_demo.py`: Dual-Camera Tabletop Demo script for the Viva Defense.
- `main_system.py`: Full production system loop (Rtsp/Dockyard integration).

### Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Add Models**:
    Place your trained `.pt` files (`plate_detection.pt`, `character_recognition.pt`) into the `models/` directory.

### Running the Research Demo
1.  Connect two webcams (Indices 1 and 2 by default).
2.  Run the demo script:
    ```bash
    python research_demo.py
    ```
3.  **Visuals**:
    - **Green Box**: Valid vehicle entry/transition.
    - **Red Box / Alert**: Speeding (<5s travel time) or Clone Attack detected.

### Running the Full System
```bash
python main_system.py
```
