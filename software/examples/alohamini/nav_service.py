import time
import zmq
import json
import threading
from navigation import NavigationController

class NavigationService:
    def __init__(self, zmq_obs_ip="127.0.0.1", zmq_obs_port=5556, zmq_cmd_port=5555):
        self.controller = NavigationController()
        
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_socket.connect(f"tcp://{zmq_obs_ip}:{zmq_obs_port}")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        
        self.pub_socket = self.context.socket(zmq.PUSH)
        self.pub_socket.connect(f"tcp://{zmq_obs_ip}:{zmq_cmd_port}")
        
        self.running = False
        self.current_pose = None
        self.rooms = {}
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("NavigationService started.")
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
            
    def go_to_room(self, room_name):
        if room_name not in self.rooms:
            print(f"[NAV] Unknown room: {room_name}")
            return False
            
        target = self.rooms[room_name]
        self.controller.set_target(target["x"], target["y"])
        return True
        
    def _loop(self):
        while self.running:
            try:
                # 1. Get Observation
                msg = self.sub_socket.recv_string()
                obs = json.loads(msg)
                
                # Update World Knowledge
                if "rooms" in obs:
                    self.rooms = obs["rooms"]
                    
                self.current_pose = {
                    "x": obs.get("x_pos", 0.0),
                    "y": obs.get("y_pos", 0.0),
                    "theta": obs.get("theta_pos", 0.0)
                }
                
                # 2. Compute Control
                action = self.controller.get_action(self.current_pose)
                
                # 3. Send Command (if active)
                if action:
                    self.pub_socket.send_string(json.dumps(action))
                    
            except Exception as e:
                print(f"[NAV] Error: {e}")
                time.sleep(0.1)

if __name__ == "__main__":
    # Test Script
    nav = NavigationService()
    nav.start()
    
    print("Waiting for connection...")
    time.sleep(2)
    
    rooms = ["Kitchen", "Bedroom", "Living Room"]
    
    for room in rooms:
        print(f"Going to {room}...")
        nav.go_to_room(room)
        
        # Wait for arrival (mock)
        time.sleep(5)
        
    nav.stop()
