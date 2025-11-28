import base64
import json
import os
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import zmq
from omni.isaac.kit import SimulationApp

# Configuration
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Show Grid
}

# Start SimulationApp
simulation_app = SimulationApp(CONFIG)

# Imports after SimulationApp
# noqa: E402
import omni.isaac.core.utils.prims as prim_utils  # noqa: E402
import omni.isaac.core.utils.stage as stage_utils  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import ArticulationSubset  # noqa: E402
from omni.isaac.core.robots import Robot  # noqa: E402
from omni.isaac.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from omni.isaac.sensor import Camera  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: F401

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(repo_root)

# Locate URDF
URDF_PATH = os.path.join(repo_root, "software/src/lerobot/robots/alohamini/alohamini.urdf")

class DatasetRecorder:
    def __init__(self, root_dir="data"):
        self.root_dir = root_dir
        self.is_recording = False
        self.current_episode_dir = None
        self.frame_idx = 0
        self.episode_idx = 0
        
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_episode_dir = os.path.join(self.root_dir, f"episode_{timestamp}")
        os.makedirs(self.current_episode_dir)
        os.makedirs(os.path.join(self.current_episode_dir, "images"))
        self.frame_idx = 0
        self.is_recording = True
        print(f"Started recording to {self.current_episode_dir}")
        
    def stop_recording(self):
        if self.is_recording:
            print(f"Stopped recording. Saved {self.frame_idx} frames.")
            self.is_recording = False
            self.current_episode_dir = None
            
    def save_frame(self, obs, action):
        if not self.is_recording:
            return
            
        # Save JSON data (state + action)
        data = {
            "timestamp": time.time(),
            "observation": {k: v for k, v in obs.items() if not isinstance(v, np.ndarray)},
            "action": action
        }
        
        with open(os.path.join(self.current_episode_dir, f"frame_{self.frame_idx:06d}.json"), "w") as f:
            json.dump(data, f)
            
        # Save Images
        for k, v in obs.items():
            if isinstance(v, np.ndarray): # Image
                img_path = os.path.join(self.current_episode_dir, "images", f"{k}_{self.frame_idx:06d}.jpg")
                cv2.imwrite(img_path, v)
                
        self.frame_idx += 1

class IsaacAlohaMini:
    def __init__(self, world, urdf_path):
        self.world = world
        self.urdf_path = urdf_path
        self.robot_prim_path = "/World/AlohaMini"
        self.robot = None
        self.cameras = {}
        
        self.setup_scene()
        
    def setup_scene(self):
        # Import URDF
        from omni.kit.commands import execute
        
        success, prim_path = execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config={
                "merge_fixed_joints": False,
                "fix_base": False,
                "make_default_prim": False,
                "create_physics_scene": True,
            },
        )
        
        if success:
            # Move to correct location
            # The importer might put it at /alohamini, we want it at /World/AlohaMini usually or just reference it
            # Let's assume it imported to the stage root with the robot name
            pass
        else:
            print(f"Failed to import URDF from {self.urdf_path}")
            sys.exit(1)

        # Find the robot prim (assuming name 'alohamini' from URDF)
        # We wrap it in an Articulation
        self.robot = Robot(prim_path="/alohamini", name="alohamini")
        self.world.scene.add(self.robot)
        
        # Add Cameras
        self.add_camera("head_front", "/alohamini/base_link/front_cam", np.array([0.2, 0, 0.2]), np.array([0, 0, 0]))
        self.add_camera("head_top", "/alohamini/base_link/top_cam", np.array([0, 0, 0.5]), np.array([0, 90, 0]))

    def add_camera(self, name, prim_path, translation, rotation_euler_deg):
        # Apply domain randomization to camera position
        # Small random perturbation to translation (+- 2cm) and rotation (+- 2 deg)
        # This helps the model become robust to slight calibration errors in the real world
        translation += np.random.uniform(-0.02, 0.02, size=3)
        rotation_euler_deg += np.random.uniform(-2, 2, size=3)
        
        # rotation in sim is usually quaternion
        # rotation_euler_deg: [x, y, z]
        rot_quat = euler_angles_to_quat(np.radians(rotation_euler_deg))
        
        camera = Camera(
            prim_path=prim_path,
            position=translation,
            frequency=30,
            resolution=(640, 480),
            orientation=rot_quat
        )
        camera.initialize()
        self.cameras[name] = camera

    def set_joint_positions(self, joint_positions: dict):
        # joint_positions: dict of joint_name -> position
        # We need to map this to the robot's dof indices or names
        # For simplicity, we can use the high level Articulation API if names match
        
        # Note: self.robot.set_joint_positions takes numpy array and indices is optional
        # We need to find indices for names
        
        current_joint_pos = self.robot.get_joint_positions()
        # This requires known order. Let's build a map once initialized
        if not hasattr(self, "dof_indices"):
            self.dof_names = [self.robot.get_dof_name(i) for i in range(self.robot.num_dof)]
            self.dof_indices = {name: i for i, name in enumerate(self.dof_names)}
            
        # Construct target array
        # Start with current to keep uncommanded joints steady
        target_pos = current_joint_pos.copy()
        
        for name, pos in joint_positions.items():
            if name in self.dof_indices:
                idx = self.dof_indices[name]
                target_pos[idx] = pos
                
        self.robot.set_joint_positions(target_pos)

    def set_base_velocity(self, vx, vy, vtheta):
        # Set root velocity
        # chassis frame: x forward, y left
        self.robot.set_linear_velocity(np.array([vx, vy, 0]))
        self.robot.set_angular_velocity(np.array([0, 0, vtheta]))

    def get_observations(self):
        obs = {}
        
        # Joints
        joint_pos = self.robot.get_joint_positions()
        if not hasattr(self, "dof_names"):
             self.dof_names = [self.robot.get_dof_name(i) for i in range(self.robot.num_dof)]
        
        for i, name in enumerate(self.dof_names):
            obs[f"{name}.pos"] = float(joint_pos[i])
            
        # Base (Ground Truth for now)
        pose = self.robot.get_world_pose()
        obs["x_pos"] = float(pose[0][0])
        obs["y_pos"] = float(pose[0][1])
        # Theta from quaternion ...
        
        # Cameras
        for name, cam in self.cameras.items():
            if cam.is_new_frame_available():
                rgba = cam.get_rgba()[:, :, :3] # Drop alpha
                # Convert to BGR for compatibility with cv2/existing pipeline if needed, 
                # but existing pipeline seems to just encode to jpg.
                # cv2 uses BGR, isaac returns RGB.
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
                obs[name] = bgr
                
        return obs

def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    aloha = IsaacAlohaMini(world, URDF_PATH)
    
    world.reset()

    # ZMQ Setup
    # Ports from standalone_sim.py
    PORT_OBS = 5556
    PORT_CMD = 5555
    
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.setsockopt(zmq.CONFLATE, 1)
    socket_pub.bind(f"tcp://*:{PORT_OBS}")
    
    socket_sub = context.socket(zmq.PULL)
    socket_sub.setsockopt(zmq.CONFLATE, 1)
    socket_sub.bind(f"tcp://*:{PORT_CMD}")
    
    print(f"Isaac Sim AlohaMini running. Ports: OBS={PORT_OBS}, CMD={PORT_CMD}")
    
    recorder = DatasetRecorder(root_dir="data_sim")
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if not world.is_playing():
            continue
            
        # 1. Receive Commands
        try:
            msg = socket_sub.recv_string(zmq.NOBLOCK)
            cmd = json.loads(msg)
            
            # Parse command
            joint_cmds = {}
            vx, vy, vth = 0, 0, 0
            
            # Check for system commands
            if "start_recording" in cmd:
                recorder.start_recording()
                continue
            if "stop_recording" in cmd:
                recorder.stop_recording()
                continue
            
            for k, v in cmd.items():
                if k == "reset" and v is True:
                     # Reset
                     print("Resetting robot...")
                     world.reset()
                     joint_cmds = {name: 0.0 for name in aloha.dof_names}
                     aloha.set_joint_positions(joint_cmds)
                     continue

                if k.endswith(".pos"):
                    joint_name = k.replace(".pos", "")
                    joint_cmds[joint_name] = v
                elif k == "x.vel":
                    vx = v
                elif k == "y.vel":
                    vy = v
                elif k == "theta.vel":
                    vth = v
                elif k == "lift_axis.height_mm":
                     joint_cmds["lift_axis"] = v
                     
            aloha.set_joint_positions(joint_cmds)
            aloha.set_base_velocity(vx, vy, vth)
            
        except zmq.Again:
            pass
        except Exception as e:
            print(f"Error receiving: {e}")

        # 2. Get Obs & Publish
        obs = aloha.get_observations()
        
        # Save frame if recording
        # (Pass current command as action label for now, though it's imperfect as it's the *commanded* not *measured* action)
        if recorder.is_recording:
            # Reconstruct action dict from parsed values
            # This is a simplification; ideally we log exactly what we sent
            action_log = {
                "x.vel": vx,
                "y.vel": vy,
                "theta.vel": vth,
                # Add arm joint targets if we had them easily accessible here
            }
            recorder.save_frame(obs, action_log)
        
        encoded_obs = {}
        
        # Process images
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                # Is image
                ret, buffer = cv2.imencode(".jpg", v, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ret:
                    encoded_obs[k] = base64.b64encode(buffer).decode("utf-8")
            else:
                encoded_obs[k] = v
                
        socket_pub.send_string(json.dumps(encoded_obs))

    simulation_app.close()

if __name__ == "__main__":
    main()
