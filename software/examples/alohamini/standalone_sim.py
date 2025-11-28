import time
import json
import base64
import argparse
import numpy as np
import zmq
import cv2
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

# --- LeKiwiSim Implementation (Adapted) ---

class LeKiwiSim:
    def __init__(self, config: LeKiwiClientConfig):
        self.config = config
        self._is_connected = False
        
        # State
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0,
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
        
        # Simulated Cameras (Noise or patterns)
        for cam, cfg in self.config.cameras.items():
            h, w = cfg.height, cfg.width
            # Create a noisy image
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            # Draw some text
            cv2.putText(img, f"{cam}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            obs[cam] = img

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
