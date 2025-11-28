import time
import json
import base64
import argparse
import numpy as np
import zmq
import cv2
import math
from dataclasses import dataclass, field

# --- Mocks for LeRobot dependencies ---

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480

@dataclass
class LeKiwiClientConfig:
    remote_ip: str = "localhost"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    teleop_keys: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.teleop_keys:
            self.teleop_keys = {
                "forward": "w", "backward": "s", "left": "a", "right": "d",
                "rotate_left": "z", "rotate_right": "x",
                "speed_up": "r", "speed_down": "f",
                "lift_up": "u", "lift_down": "j", "quit": "q",
            }
        if not self.cameras:
            self.cameras = {
                "head_front": CameraConfig(),
                "head_top": CameraConfig() # Add more if needed
            }

class LiftAxisConfig:
    step_mm = 1.0

class DeviceNotConnectedError(Exception):
    pass

# --- Simulation Classes ---

@dataclass
class SimObject:
    label: str
    x: float  # meters
    y: float  # meters
    width: float = 0.2 # meters
    height: float = 0.2 # meters
    color: tuple = (0, 0, 255) # BGR
    shape: str = "rectangle" # or circle

# --- LeKiwiSim Implementation (Adapted) ---

class LeKiwiSim:
    def __init__(self, config: LeKiwiClientConfig):
        self.config = config
        self._is_connected = False
        
        # State
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0, # degrees
        }
        
        # World Objects
        self.objects = [
            SimObject("soda_can", 1.0, 0.5, 0.1, 0.1, (255, 0, 0), "circle"), # Blue Soda
            SimObject("trash", 1.5, -0.5, 0.2, 0.2, (0, 0, 255), "rectangle"), # Red Trash
            SimObject("charger", 2.0, 0.0, 0.3, 0.3, (0, 255, 0), "rectangle"), # Green Charger
            SimObject("table", 0.0, 1.5, 0.8, 0.4, (42, 42, 165), "rectangle"), # Brownish Table
        ]

        # Semantic Map (Room coordinates)
        self.rooms = {
            "Living Room": {"x": 0.0, "y": 0.0},
            "Kitchen": {"x": 2.0, "y": 2.0},
            "Bedroom": {"x": -2.0, "y": 1.0},
            "Hallway": {"x": 0.0, "y": 3.0}
        }

        # Joint positions
        self.joints = {}
        joint_names = [
                "arm_left_shoulder_pan", "arm_left_shoulder_lift", "arm_left_elbow_flex",
                "arm_left_wrist_flex", "arm_left_wrist_roll", "arm_left_gripper",
                "arm_right_shoulder_pan", "arm_right_shoulder_lift", "arm_right_elbow_flex",
                "arm_right_wrist_flex", "arm_right_wrist_roll", "arm_right_gripper",
        ]
        self.joints.update({k: 0.0 for k in joint_names})
        self.joints["lift_axis"] = 0.0
        
        self.teleop_keys = config.teleop_keys
        self.speed_levels = [
            {"xy": 0.15, "theta": 45},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.25, "theta": 75},
        ]
        self.speed_index = 0
        self.last_update = time.perf_counter()
        
        # Current velocities for observation
        self.current_vel = {"x": 0.0, "y": 0.0, "theta": 0.0}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self):
        print("LeKiwiSim connected.")
        self._is_connected = True

    def disconnect(self):
        print("LeKiwiSim disconnected.")
        self._is_connected = False
        
    def _world_to_robot(self, wx, wy):
        """Convert world coordinates to robot-centric coordinates (robot is at 0,0 facing +X)"""
        dx = wx - self.state["x"]
        dy = wy - self.state["y"]
        
        # Rotate by -theta
        rad = np.radians(self.state["theta"])
        c, s = np.cos(rad), np.sin(rad)
        
        # Standard rotation matrix for rotating points CCW is:
        # x' = x cos - y sin
        # y' = x sin + y cos
        # Here we rotate the *points* by -theta (or coordinate system by +theta)
        # So we use -theta.
        # x' = dx * cos(-t) - dy * sin(-t) = dx * c + dy * s
        # y' = dx * sin(-t) + dy * cos(-t) = -dx * s + dy * c
        
        rx = dx * c + dy * s
        ry = -dx * s + dy * c
        
        return rx, ry

    def render_camera(self, cam_name, width, height):
        # Create a background (Gray floor)
        img = np.full((height, width, 3), 200, dtype=np.uint8)
        
        # Grid lines (every 1 meter)
        # We need to map world coords to pixel coords.
        # Let's say: Center of image is Robot (0,0 local).
        # Scale: 100 pixels = 1 meter.
        scale = 100.0
        cx = width // 2
        cy = height // 2
        
        # Draw grid
        # We iterate a range around the robot's world position
        rx, ry = self.state["x"], self.state["y"]
        min_gx = int(rx - width/scale/2) - 1
        max_gx = int(rx + width/scale/2) + 1
        min_gy = int(ry - height/scale/2) - 1
        max_gy = int(ry + height/scale/2) + 1
        
        for gx in range(min_gx, max_gx + 1):
             # Transform world grid line x=gx to robot space, then to pixels
             p1 = self._world_to_robot(gx, ry - 10) # Line stretches far
             p2 = self._world_to_robot(gx, ry + 10)
             
             # Project to pixels: x_pix = cx + x_rob * scale, y_pix = cy - y_rob * scale (since image y is down)
             # Note: In robot frame, +X is forward? No, usually +X is forward in ROS, but here I used +X is East, Theta is CCW from East.
             # Let's assume Robot "Forward" is aligned with its Theta.
             # So in "_world_to_robot", rx is distance "Forward" (along heading), ry is distance "Left" (cross product).
             # Let's check math in _world_to_robot:
             # If theta=0, rx=dx, ry=dy. Robot facing East. Obj at (1,0) -> rx=1 (Forward), ry=0.
             # If theta=90, rx=dy, ry=-dx. Robot facing North. Obj at (0,1) -> rx=1 (Forward), ry=0.
             # Yes, rx is "Forward", ry is "Left".
             
             # So: Screen UP should be Robot Forward (rx). Screen RIGHT should be Robot Right (-ry).
             # Screen X = cx - ry * scale
             # Screen Y = cy - rx * scale
             
             pt1_screen = (int(cx - p1[1] * scale), int(cy - p1[0] * scale))
             pt2_screen = (int(cx - p2[1] * scale), int(cy - p2[0] * scale))
             cv2.line(img, pt1_screen, pt2_screen, (180, 180, 180), 1)

        for gy in range(min_gy, max_gy + 1):
             p1 = self._world_to_robot(rx - 10, gy)
             p2 = self._world_to_robot(rx + 10, gy)
             pt1_screen = (int(cx - p1[1] * scale), int(cy - p1[0] * scale))
             pt2_screen = (int(cx - p2[1] * scale), int(cy - p2[0] * scale))
             cv2.line(img, pt1_screen, pt2_screen, (180, 180, 180), 1)

        # Draw Robot (Triangle)
        # Robot is at 0,0 in robot frame.
        robot_pts = np.array([
            [0, -10], # Back Center (slightly back) -> wait.
            # Forward is +rx -> Up on screen (-y).
            # Right is -ry -> Right on screen (+x).
            # Triangle pointing up:
            [cx, cy - 20], # Tip (Forward)
            [cx - 15, cy + 15], # Back Left
            [cx + 15, cy + 15]  # Back Right
        ], np.int32)
        cv2.fillPoly(img, [robot_pts], (50, 50, 50))
        
        detections = []

        # Draw Objects
        for obj in self.objects:
            # Transform to robot frame
            rx, ry = self._world_to_robot(obj.x, obj.y)
            
            # Project to screen
            sx = int(cx - ry * scale)
            sy = int(cy - rx * scale)
            
            # Check if roughly in view
            if 0 <= sx < width and 0 <= sy < height:
                # Size in pixels
                w_pix = int(obj.width * scale)
                h_pix = int(obj.height * scale)
                
                # Bounding box for object
                top_left = (sx - w_pix // 2, sy - h_pix // 2)
                bottom_right = (sx + w_pix // 2, sy + h_pix // 2)
                
                # Draw
                if obj.shape == "circle":
                    cv2.circle(img, (sx, sy), w_pix // 2, obj.color, -1)
                else:
                    cv2.rectangle(img, top_left, bottom_right, obj.color, -1)
                
                # Add label
                cv2.putText(img, obj.label, (sx - w_pix//2, sy - h_pix//2 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                
                # Create Detection (Perfect Detector)
                # Bounding box in pixel coordinates [x, y, w, h] or [x1, y1, x2, y2]
                # Let's use [x1, y1, x2, y2]
                # Clip to screen
                x1 = max(0, top_left[0])
                y1 = max(0, top_left[1])
                x2 = min(width, bottom_right[0])
                y2 = min(height, bottom_right[1])
                
                if x2 > x1 and y2 > y1:
                    detections.append({
                        "label": obj.label,
                        "confidence": 1.0,
                        "box": [x1, y1, x2, y2] # Pixel coords
                    })

        return img, detections

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError("Not connected")

        obs = {}
        for k, v in self.joints.items():
            if k == "lift_axis": continue
            obs[f"{k}.pos"] = v
            
        # Return actual simulated velocities
        obs["x.vel"] = self.current_vel["x"]
        obs["y.vel"] = self.current_vel["y"]
        obs["theta.vel"] = self.current_vel["theta"]
        
        # Also return Pose for visualization
        obs["x_pos"] = self.state["x"]
        obs["y_pos"] = self.state["y"]
        obs["theta_pos"] = self.state["theta"]
        
        obs["lift_axis.height_mm"] = self.joints["lift_axis"]
        
        # Simulated Perception
        obs["detections"] = {}
        obs["rooms"] = self.rooms # Broadcast room locations too
        
        for cam, cfg in self.config.cameras.items():
            img, dets = self.render_camera(cam, cfg.width, cfg.height)
            obs[cam] = img
            obs["detections"][cam] = dets

        return obs

    def send_action(self, action: dict) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError("Not connected")
            
        now = time.perf_counter()
        dt = now - self.last_update
        self.last_update = now
        
        vx = action.get("x.vel", 0.0)
        vy = action.get("y.vel", 0.0)
        vth = action.get("theta.vel", 0.0)
        
        # Store for observation
        self.current_vel["x"] = vx
        self.current_vel["y"] = vy
        self.current_vel["theta"] = vth
        
        rad = np.radians(self.state["theta"])
        c, s = np.cos(rad), np.sin(rad)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
        dth = vth * dt
        
        self.state["x"] += dx
        self.state["y"] += dy
        self.state["theta"] += dth
        
        for k, v in action.items():
            if k.endswith(".pos"):
                joint = k.replace(".pos", "")
                self.joints[joint] = v
            elif k == "lift_axis.height_mm":
                self.joints["lift_axis"] = v
                
        return action
    
    def stop_base(self):
        pass


# --- Host Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config = LeKiwiClientConfig()
    robot = LeKiwiSim(config)
    robot.connect()

    # ZMQ Setup
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.setsockopt(zmq.CONFLATE, 1)
    socket_pub.bind(f"tcp://*:{config.port_zmq_observations}")
    
    socket_sub = context.socket(zmq.PULL)
    socket_sub.setsockopt(zmq.CONFLATE, 1)
    socket_sub.bind(f"tcp://*:{config.port_zmq_cmd}")

    print(f"Simulator running on ports: OBS={config.port_zmq_observations}, CMD={config.port_zmq_cmd}")

    try:
        while True:
            start_time = time.perf_counter()
            
            # 1. Process Commands (Non-blocking)
            try:
                msg = socket_sub.recv_string(zmq.NOBLOCK)
                action = json.loads(msg)
                robot.send_action(action)
            except zmq.Again:
                pass
            except Exception as e:
                print(f"Error receiving command: {e}")

            # 2. Get Observation
            obs = robot.get_observation()
            
            # 3. Encode Images
            encoded_obs = obs.copy()
            
            # Remove raw image data from encoded_obs to save bandwidth/processing if not needed, 
            # but we need to encode it first.
            
            # Special handling for detections: keep them as object
            # (they are already json serializable)
            
            for cam in config.cameras:
                if cam in obs:
                    ret, buffer = cv2.imencode(".jpg", obs[cam], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ret:
                        encoded_obs[cam] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        encoded_obs[cam] = ""
                        
            # 4. Publish
            socket_pub.send_string(json.dumps(encoded_obs))
            
            # 5. Sleep
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / args.fps) - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        robot.disconnect()
        socket_pub.close()
        socket_sub.close()
        context.term()
        
if __name__ == "__main__":
    main()
