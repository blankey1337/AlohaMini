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
    "headless": True,
    # "renderer": "RayTracedLighting", # Disable explicit RT to fallback to default (Raster) for stability
    "display_options": 3286,  # Show Grid
    "livesync_usd": None,
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
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from omni.isaac.core.utils.nucleus import get_assets_root_path  # noqa: E402
from omni.isaac.sensor import Camera  # noqa: E402
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCylinder, FixedCuboid # noqa: E402
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
        self.camera_prims = {}
        self.camera_pitch = 0.0 # deg
        
        self.setup_scene()
        self.setup_environment()
        
    def setup_environment(self):
        # Try to load a realistic environment from Nucleus
        assets_root_path = get_assets_root_path()
        env_loaded = False
        
        if assets_root_path:
            # Common realistic environments
            # simple_room_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
            # Or a modern office
            env_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
            
            try:
                # We use add_reference_to_stage to load the USD into the current stage
                stage_utils.add_reference_to_stage(env_path, "/World/Environment")
                print(f"Loaded environment from {env_path}")
                env_loaded = True
                
                # Move the environment so the robot (at 0,0) isn't inside a table
                # The Simple Room often has a table at the center
                env_prim = XFormPrim("/World/Environment")
                env_prim.set_world_pose(translation=np.array([0.0, 2.0, 0.0])) # Move room 2m to the side
                
            except Exception as e:
                print(f"Failed to load environment asset: {e}")
    
        if not env_loaded:
            print("Nucleus assets not found or failed to load. Using procedural environment.")
            # Fallback: Add ground plane and walls
            self.world.scene.add_default_ground_plane()
            
            wall_back = FixedCuboid(
                prim_path="/World/WallBack",
                name="wall_back",
                position=np.array([-0.5, 0.0, 1.0]),
                scale=np.array([0.1, 3.0, 2.0]),
                color=np.array([0.9, 0.9, 0.9])
            )
            self.world.scene.add(wall_back)

        # Always add the manipulation task elements (Desk + Objects)
        # We might need to adjust their position if the room has walls at 0,0
        
        # 1. Desk (In front of robot)
        # Shift desk forward (positive x) so robot doesn't spawn inside it or too close
        # Lower the desk to ~15cm height so it's visible to the low-mounted camera
        desk_height = 0.15
        desk_z = desk_height / 2.0
        
        desk = FixedCuboid(
            prim_path="/World/Desk",
            name="desk",
            position=np.array([1.5, 0.0, desk_z]), # Moved from 1.0 to 1.5
            scale=np.array([0.5, 1.0, desk_height]), 
            color=np.array([0.4, 0.3, 0.2]) 
        )
        self.world.scene.add(desk)
        
        # 2. Objects on Desk
        # Ball
        ball = DynamicSphere(
            prim_path="/World/Ball",
            name="ball",
            position=np.array([1.5, 0.2, desk_height + 0.04]), # Adjusted z to sit on desk
            radius=0.04,
            color=np.array([0.8, 0.1, 0.1]),
            mass=0.1
        )
        self.world.scene.add(ball)
        
        # Notepad
        notepad = DynamicCuboid(
            prim_path="/World/Notepad",
            name="notepad",
            position=np.array([1.5, -0.2, desk_height + 0.01]), # Adjusted z
            scale=np.array([0.15, 0.2, 0.02]),
            color=np.array([0.9, 0.9, 0.9]),
            mass=0.1
        )
        self.world.scene.add(notepad)

        # Pencil
        pencil = DynamicCylinder(
            prim_path="/World/Pencil",
            name="pencil",
            position=np.array([1.5, -0.2, desk_height + 0.02]), # Adjusted z
            radius=0.005,
            height=0.15,
            color=np.array([0.9, 0.8, 0.1]),
            mass=0.02,
            orientation=euler_angles_to_quat(np.radians(np.array([0, 90, 0]))) # Rotate to lay flat
        )
        self.world.scene.add(pencil)
        
        # 3. Walls (Moved to fallback)
        
        # Add a DistantLight to ensure the scene is lit even if Environment fails or is dark
        # This is critical for headless rendering without RayTracing
        from omni.isaac.core.prims import XFormPrim
        from pxr import UsdLux
        
        # Create a distant light
        stage = self.world.stage
        light_prim_path = "/World/defaultLight"
        if not stage.GetPrimAtPath(light_prim_path):
            light = UsdLux.DistantLight.Define(stage, light_prim_path)
            light.CreateIntensityAttr(3000) # High intensity for raster
            light.CreateAngleAttr(0.53)
            light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            
            # Orient it to shine down and forward
            xform = UsdGeom.Xformable(light)
            # Rotate around X to shine down
            xform.AddRotateXOp().Set(-60)


    def setup_scene(self):
        # Import URDF using omni.importer.urdf API directly for 4.x compatibility
        # The command interface changed between 2023.x and 4.x
        
        import omni.kit.commands
        
        # In Isaac Sim 4.0+, we can often use the URDFImporter extension directly 
        # or use the command with updated arguments.
        # But to be robust, let's use the extension interface if possible, or try the command with correct args.
        
        # Try the command with the new signature if known, or use the lower level API.
        # The error showed: import_robot(self, assetRoot, assetName, robot, importConfig, ...)
        # We need a UrdfRobot object if we use that low level function.
        
        # Easiest path: Use the high-level 'URDFParseAndImportFile' command but ensure arguments are correct.
        # The error suggests the command wrapper is failing to map arguments correctly.
        
        # Let's try direct Extension API usage which bypasses the Command wrapper ambiguity.
        from omni.importer.urdf import _urdf
        
        urdf_interface = _urdf.acquire_urdf_interface()
        
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.fix_base = False
        import_config.make_default_prim = False
        import_config.create_physics_scene = True
        
        root_path = os.path.dirname(self.urdf_path)
        file_name = os.path.basename(self.urdf_path)
        
        # 1. Parse
        imported_robot = urdf_interface.parse_urdf(root_path, file_name, import_config)
        
        # 2. Import
        dest_path = "" # Import to root or default prim
        prim_path = urdf_interface.import_robot(root_path, file_name, imported_robot, import_config, dest_path)
        
        if prim_path:
            pass
        else:
            print(f"Failed to import URDF from {self.urdf_path}")
            sys.exit(1)

        # Find the robot prim (assuming name 'alohamini' from URDF)
        # We wrap it in an Articulation
        self.robot = Robot(prim_path="/alohamini", name="alohamini")
        
        # Explicitly set the robot position to avoid floor clipping
        # Wheels are at Z=0.05 with radius 0.05, so Z=0 should be fine, but let's lift it 1cm to be safe.
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.01]))
        
        self.world.scene.add(self.robot)
        
        # Add Cameras
        # 1. Robot-mounted
        self.add_camera("head_front", "/alohamini/base_link/front_cam", np.array([0.2, 0, 0.2]), np.array([0., 0., 0.]))
        # Top cam: Positioned higher (0.8m) and looking down (90 deg pitch)
        self.add_camera("head_top", "/alohamini/base_link/top_cam", np.array([0, 0, 0.8]), np.array([0., 90., 0.]))
        
        # 2. Third-person (World fixed)
        # Back view (Driver's view)
        self.add_camera("cam_high_back", "/World/CamBack", np.array([-1.2, 0, 1.2]), np.array([0., 25., 0.]), resolution=(640, 480))
        # Front view (Looking at robot)
        self.add_camera("cam_front_view", "/World/CamFront", np.array([2.5, 0, 1.0]), np.array([0., 15., 180.]), resolution=(640, 480))

    def configure_controllers(self):
        # Configure joint drives (stiffness/damping)
        # Especially for lift_axis to prevent falling
        # This must be done after the robot is added to the scene and reset
        
        # Initialize the robot view to ensure internal handles are created
        self.robot.initialize()

        # We need dof indices to set gains correctly
        # Initialize indices immediately
        # Check if num_dof is valid (it might be 0 if not initialized)
        if self.robot.num_dof > 0:
            if hasattr(self.robot, "dof_names"):
                # Isaac Sim 4.x
                self.dof_names = self.robot.dof_names
            else:
                 # Older versions
                 self.dof_names = [self.robot.get_dof_name(i) for i in range(self.robot.num_dof)]
            
            self.dof_indices = {name: i for i, name in enumerate(self.dof_names)}
            
            # Default gains
            kps = np.ones(self.robot.num_dof) * 1000.0
            kds = np.ones(self.robot.num_dof) * 100.0
            
            # Higher gains for lift to fight gravity
            if "lift_axis" in self.dof_indices:
                lift_idx = self.dof_indices["lift_axis"]
                kps[lift_idx] = 10000.0
                kds[lift_idx] = 1000.0
            
            # Try to set gains using the ArticulationController (standard way)
            try:
                controller = self.robot.get_articulation_controller()
                controller.set_gains(kps, kds)
            except Exception as e:
                print(f"Failed to set gains via ArticulationController: {e}")
                print("Available methods on Robot:", dir(self.robot))
                # Attempt direct property setting if controller fails (fallback for some versions)
                try:
                    # Check for direct attribute access (PhysX bindings)
                    # This is highly version dependent
                    pass 
                except:
                    pass
        else:
            print("Warning: Robot has 0 DOFs or is not initialized correctly.")
            self.dof_names = []
            self.dof_indices = {}

    def add_camera(self, name, prim_path, translation, rotation_euler_deg, resolution=(640, 480)):
        # Ensure float type for in-place addition
        rotation_euler_deg = rotation_euler_deg.astype(float)
        
        # Apply domain randomization to camera position
        # Small random perturbation to translation (+- 2cm) and rotation (+- 2 deg)
        # This helps the model become robust to slight calibration errors in the real world
        # Only randomize if it's a robot camera (heuristic: name starts with head)
        if name.startswith("head"):
            translation += np.random.uniform(-0.02, 0.02, size=3)
            rotation_euler_deg += np.random.uniform(-2, 2, size=3)
        
        # rotation in sim is usually quaternion
        # rotation_euler_deg: [x, y, z]
        rot_quat = euler_angles_to_quat(np.radians(rotation_euler_deg))
        
        camera = Camera(
            prim_path=prim_path,
            position=translation,
            frequency=30,
            resolution=resolution,
            orientation=rot_quat
        )
        camera.initialize()
        
        # Set clipping range to avoid near plane clipping objects
        # Especially critical for the front camera which is close to the desk
        # Default is often 0.1 or 1.0 which is too large for manipulation
        camera.set_clipping_range(0.01, 100.0)
        
        self.cameras[name] = camera
        
        # We used to store XFormPrim here for cheating camera pose, but we removed it
        # because the real robot has fixed cameras.
        # self.camera_prims[name] = XFormPrim(prim_path)

    # Removed set_camera_pose because real robot cameras are fixed.
    # To fix orientation issues, we should adjust the rotation in add_camera.

    def set_joint_positions(self, joint_positions: dict):
        # joint_positions: dict of joint_name -> position
        # We need to map this to the robot's dof indices or names
        # For simplicity, we can use the high level Articulation API if names match
        
        # Note: self.robot.set_joint_positions takes numpy array and indices is optional
        # We need to find indices for names
        
        current_joint_pos = self.robot.get_joint_positions()
        # This requires known order. Let's build a map once initialized
        if not hasattr(self, "dof_indices"):
            if hasattr(self.robot, "dof_names"):
                self.dof_names = self.robot.dof_names
            else:
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
        # Set root velocity in ROBOT frame
        # We need to rotate vx, vy by the robot's current yaw
        
        # 1. Get current pose
        pose = self.robot.get_world_pose()
        # pose[0] is position, pose[1] is quaternion [w, x, y, z]
        quat = pose[1]
        
        # 2. Convert to rotation matrix or just get yaw
        # A simple way for 2D rotation:
        # q = [w, x, y, z]
        w, x, y, z = quat
        # Yaw (rotation around Z) calculation from quaternion
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # 3. Rotate velocity vector
        # World VX = vx * cos(yaw) - vy * sin(yaw)
        # World VY = vx * sin(yaw) + vy * cos(yaw)
        
        world_vx = vx * np.cos(yaw) - vy * np.sin(yaw)
        world_vy = vx * np.sin(yaw) + vy * np.cos(yaw)
        
        self.robot.set_linear_velocity(np.array([world_vx, world_vy, 0]))
        self.robot.set_angular_velocity(np.array([0, 0, vtheta]))

    def get_observations(self):
        obs = {}
        
        # Joints
        joint_pos = self.robot.get_joint_positions()
        if not hasattr(self, "dof_names"):
             # For Isaac Sim 4.x, dof_names is a property, usually on the Articulation view
             if hasattr(self.robot, "dof_names"):
                 self.dof_names = self.robot.dof_names
             else:
                 # Fallback for older versions or if wrapped differently
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
            # In Isaac Sim 4.x, just try to get the frame.
            # Some older methods like is_new_frame_available are deprecated/removed.
            
            # Use get_rgba() directly. It should return None or old frame if not ready.
            # Or better, check render_product availability if needed, but simple access is usually safe.
            try:
                rgba = cam.get_rgba()[:, :, :3] # Drop alpha
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
                obs[name] = bgr
            except Exception:
                # Frame might not be ready yet
                pass
                
        return obs

def main():
    world = World(stage_units_in_meters=1.0)
    # world.scene.add_default_ground_plane()  # Moved to setup_environment to avoid conflict with loaded rooms
    
    aloha = IsaacAlohaMini(world, URDF_PATH)
    
    world.reset()
    
    # Configure after reset to ensure physics handles are valid
    aloha.configure_controllers()

    # ZMQ Setup
    # Ports from standalone_sim.py
    PORT_OBS = 5556
    PORT_CMD = 5555
    
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.setsockopt(zmq.CONFLATE, 1)
    socket_pub.bind(f"tcp://0.0.0.0:{PORT_OBS}")
    
    socket_sub = context.socket(zmq.PULL)
    socket_sub.setsockopt(zmq.CONFLATE, 1)
    socket_sub.bind(f"tcp://0.0.0.0:{PORT_CMD}")
    
    print(f"Isaac Sim AlohaMini running. Ports: OBS={PORT_OBS}, CMD={PORT_CMD}")
    sys.stdout.flush()
    
    recorder = DatasetRecorder(root_dir="data_sim")
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if world.current_time_step_index % 100 == 0:
             print(f"Sim Step: {world.current_time_step_index}, Time: {world.current_time}")
        
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
                     aloha.configure_controllers() # Re-apply gains
                     joint_cmds = {name: 0.0 for name in aloha.dof_names}
                     aloha.set_joint_positions(joint_cmds)
                     continue

                if k.endswith(".pos"):
                    joint_name = k.replace(".pos", "")
                    joint_cmds[joint_name] = v
                    
                    # Mirror gripper commands for the second finger
                    if joint_name == "arm_left_gripper":
                         joint_cmds["arm_left_gripper_2"] = v
                    elif joint_name == "arm_right_gripper":
                         joint_cmds["arm_right_gripper_2"] = v
                elif k == "x.vel":
                    vx = v
                elif k == "y.vel":
                    vy = v
                elif k == "theta.vel":
                    vth = v
                elif k == "lift_axis.height_mm":
                     # Convert mm to meters
                     joint_cmds["lift_axis"] = v / 1000.0
                     
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
                # Debug print for black image check
                if np.max(v) == 0:
                     print(f"Warning: Camera {k} is returning all black pixels!")
                
                ret, buffer = cv2.imencode(".jpg", v, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ret:
                    encoded_obs[k] = base64.b64encode(buffer).decode("utf-8")
            else:
                encoded_obs[k] = v
                
        socket_pub.send_string(json.dumps(encoded_obs))

    simulation_app.close()

if __name__ == "__main__":
    main()
