from datetime import datetime
from collections import defaultdict

class SpatialVerifier:
    def __init__(self, travel_times):
        """
        travel_times: dict like {('GATE1', 'GATE2'): 120}  # seconds
        """
        self.travel_times = travel_times
        self.vehicle_state = {}  # {plate: {'gate': 'GATE1', 'timestamp': datetime}}
        self.fraud_alerts = []
    
    def check_entry(self, plate_number, gate_id):
        """
        Modified for Tabletop Research Demo:
        - Detects SPEEDING / TELEPORTATION (Clone)
        - Detects PATH VIOLATION (Jump)
        """
        now = datetime.now()
        
        # 1. New Vehicle (Entry to System)
        if plate_number not in self.vehicle_state:
            self.vehicle_state[plate_number] = {
                'gate': gate_id,
                'timestamp': now
            }
            return (True, "System Entry")
        
        prev = self.vehicle_state[plate_number]
        prev_gate = prev['gate']
        prev_time = prev['timestamp']
        
        # 2. Same Gate Update
        if prev_gate == gate_id:
            # Simple update - no transition check needed
            # For demo, if same gate, just update timestamp
            self.vehicle_state[plate_number]['timestamp'] = now
            return (True, "Same Gate")

        # 3. Transition Analysis
        route_key = (prev_gate, gate_id)
        reverse_key = (gate_id, prev_gate)
        
        min_required_time = self.travel_times.get(route_key)
        # Check reverse if not defined (assume bidirectional same time for simplicity unless specified)
        if min_required_time is None:
             min_required_time = self.travel_times.get(reverse_key)

        # Calculate actual travel time
        actual_time = (now - prev_time).total_seconds()

        # Scenario: Path Violation (No direct connection or skipped checkpoint)
        # In a graph, if A->C exists but requires B, then (A,C) is not in travel_times directly, or has a huge time?
        if min_required_time is None:
            # If no edge exists at all:
            reason = f"PATH VIOLATION: {prev_gate} -> {gate_id} (No direct path)"
            self._log_fraud(plate_number, prev_gate, gate_id, "N/A", actual_time, reason)
            # Update state despite violation to track current location
            self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now} 
            return (False, reason)

        # Scenario: Speeding / Clone Attack
        if actual_time < min_required_time:
            # Threshold for "Clone" vs "Speeding"
            # If time is extremely low (e.g. < 2s on a 120s route), it's physically impossible = Teleportation/Clone
            
            if actual_time < 0.5: 
                reason = f"CLONE ATTACK: {prev_gate}->{gate_id} in {actual_time:.2f}s (Min {min_required_time}s)"
            else:
                reason = f"PHYSICS VIOLATION: {prev_gate}->{gate_id} in {actual_time:.2f}s (Min {min_required_time}s)"
            
            self._log_fraud(plate_number, prev_gate, gate_id, min_required_time, actual_time, reason)
            
            # Crucial for clone attack: The system now bans this *new* appearance? 
            # Or flags it and updates? Usually we flag the anomaly.
            self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now}
            return (False, reason)
        
        # 4. Valid Transition
        self.vehicle_state[plate_number] = {'gate': gate_id, 'timestamp': now}
        return (True, f"Valid: {actual_time:.1f}s")
    
    def _log_fraud(self, plate, origin, dest, limit, actual, reason):
        alert = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'plate': plate,
            'reason': reason
        }
        self.fraud_alerts.append(alert)
        # Print immediately for demo effect
        print(f"\n🚨  [SECURITY ALERT] {plate}: {reason}")

    def get_fraud_report(self):
        return self.fraud_alerts
