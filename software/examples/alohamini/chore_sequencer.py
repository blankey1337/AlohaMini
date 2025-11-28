import time
import json
import threading
import zmq
from nav_service import NavigationService

class ChoreExecutor:
    def __init__(self):
        self.nav = NavigationService()
        self.nav.start()
        
        self.running = False
        self.current_chore = None
        self.chore_thread = None
        
        # Robot State
        self.detections = {}
        
        # Subscribe to obs for detections
        self.nav.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        # We need a way to read detections. NavigationService already reads obs.
        # Let's piggyback or just read from nav service state?
        # Nav service only stores pose and rooms.
        # Let's subclass or monkeypatch NavService to expose detections, 
        # OR just read from the socket in NavService and store it.
        # Actually, let's just modify NavService to store full obs or detections.
        
    def start_chore(self, chore_name):
        if self.running:
            print("[CHORE] Already running a chore.")
            return
            
        self.running = True
        self.current_chore = chore_name
        self.chore_thread = threading.Thread(target=self._run_chore, args=(chore_name,))
        self.chore_thread.start()
        
    def stop_chore(self):
        self.running = False
        if self.chore_thread:
            self.chore_thread.join()
        self.nav.stop()
        
    def _run_chore(self, chore_name):
        print(f"[CHORE] Starting: {chore_name}")
        
        if chore_name == "clean_up":
            self._do_clean_up()
        else:
            print(f"[CHORE] Unknown chore: {chore_name}")
            
        self.running = False
        print(f"[CHORE] Finished: {chore_name}")

    def _do_clean_up(self):
        # 1. Go to Kitchen
        print("[CHORE] Step 1: Go to Kitchen")
        self.nav.go_to_room("Kitchen")
        self._wait_until_idle() 
        
        # 2. Look for Trash
        print("[CHORE] Step 2: Scan for Trash")
        trash = self._wait_for_detection("trash")
        
        if trash:
            print(f"[CHORE] Found trash at {trash.get('box')}!")
            # 3. Pick it up (Mock action)
            print("[CHORE] Step 3: Picking up trash...")
            time.sleep(2)
            
            # 4. Go to Bin (Let's say Bin is in Hallway)
            print("[CHORE] Step 4: Go to Hallway (Trash Bin)")
            self.nav.go_to_room("Hallway")
            self._wait_until_idle()
            
            # 5. Drop it
            print("[CHORE] Step 5: Dropping trash")
            time.sleep(1)
        else:
            print("[CHORE] No trash found.")

    def _wait_for_detection(self, label, timeout=5.0):
        # Poll self.nav.detections
        start = time.time()
        while time.time() - start < timeout:
            dets = self.nav.detections
            for cam, items in dets.items():
                for item in items:
                    if item.get("label") == label:
                        return item
            time.sleep(0.1)
        return None

    def _wait_until_idle(self):
        # Wait for nav service to report idle
        # (It might briefly be idle before starting move, so wait a tiny bit first if needed)
        time.sleep(0.5) 
        while not self.nav.is_idle():
            time.sleep(0.1)

if __name__ == "__main__":
    executor = ChoreExecutor()
    
    # Wait for connection
    time.sleep(2)
    
    print("Available Rooms:", executor.nav.rooms.keys())
    
    executor.start_chore("clean_up")
    
    # Keep main thread alive
    while executor.running:
        time.sleep(1)
        
    executor.stop_chore()
