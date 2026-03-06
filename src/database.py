import queue
import threading
import requests

class DatabaseManager:
    def __init__(self, base_url, cam_code, device, in_out):
        self.base_url = base_url
        self.cam_code = cam_code
        self.device = device
        self.in_out = in_out
        self.insert_endpoint = f"{base_url}/PostRFID"
        self.select_endpoint = f"{base_url}/GetCamDetails"
        
        self.insert_queue = queue.Queue()
        self.is_running = True
        self.insert_thread = threading.Thread(target=self._process_insert_queue, daemon=True)
        self.insert_thread.start()
        
        self.stats = {'total_inserts': 0, 'successful_inserts': 0, 'failed_inserts': 0}
        print(f"🗄️  Database Manager initialized - Camera {device} ({in_out})")
    
    def insert_plate_detection(self, plate_number, confidence, track_id, sync=False):
        detection_data = {'plate_number': plate_number}
        
        if sync:
            return self._perform_insert(detection_data)
        else:
            self.insert_queue.put(detection_data)
            return True
    
    def _perform_insert(self, detection_data):
        try:
            params = {
                'CamCode': detection_data['plate_number'],
                'Device': self.device,
                'InOut': self.in_out
            }
            
            response = requests.get(self.insert_endpoint, params=params, timeout=10)
            self.stats['total_inserts'] += 1
            
            if response.status_code in [200, 201]:
                self.stats['successful_inserts'] += 1
                print(f"✅ Plate '{detection_data['plate_number']}' saved to database")
                return True
            else:
                self.stats['failed_inserts'] += 1
                return False
        except Exception as e:
            self.stats['failed_inserts'] += 1
            print(f"❌ Database error: {e}")
            return False
    
    def _process_insert_queue(self):
        while self.is_running:
            try:
                detection_data = self.insert_queue.get(timeout=1)
                self._perform_insert(detection_data)
                self.insert_queue.task_done()
            except queue.Empty:
                continue
    
    def get_stats(self):
        return self.stats.copy()
    
    def shutdown(self):
        self.is_running = False
        if self.insert_thread.is_alive():
            self.insert_thread.join(timeout=2)
