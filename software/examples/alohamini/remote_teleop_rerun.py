import argparse
import base64
import json
import select
import sys
import termios
import threading
import time
import tty
import os

import cv2
import numpy as np
import zmq

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(repo_root)

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
import rerun as rr

# --- Config ---
CMD_PORT = 5555
OBS_PORT = 5556
DEFAULT_IP = "127.0.0.1"

msg = """
Remote Teleop & Rerun Visualization
-----------------------------------
Moving around:
   q    w    e
   a    s    d
   z    x    c

w/x : increase/decrease linear x speed (Forward/Backward)
a/d : increase/decrease linear y speed (Left/Right)
q/e : increase/decrease angular speed (Rotate Left/Right)

u/j : increase/decrease lift height

Arm Control (Left):
z/x : Shoulder Pan (+/-)
c/v : Shoulder Lift (+/-)
b/n : Elbow Flex (+/-)
m/, : Gripper (Open/Close)

r   : Toggle Recording (Start/Stop)

space, k : force stop
CTRL-C to quit
"""

# Key mappings
MOVE_BINDINGS = {
    'w': ('x.vel', 0.05),
    's': ('x.vel', -0.05),
    'a': ('y.vel', 0.05),
    'd': ('y.vel', -0.05),
    'q': ('theta.vel', 1.0),  # Reduced rotation speed for better control
    'e': ('theta.vel', -1.0),
}

LIFT_BINDINGS = {
    'u': ('lift_axis.height_mm', 2.0),
    'j': ('lift_axis.height_mm', -2.0),
}

# Arm Control (Left Arm mainly for now)
# Joint names from URDF:
# arm_left_shoulder_pan, arm_left_shoulder_lift, arm_left_elbow_flex, 
# arm_left_wrist_flex, arm_left_wrist_roll, arm_left_gripper
ARM_BINDINGS = {
    # Shoulder Pan
    'z': ('arm_left_shoulder_pan.pos', 0.1),
    'x': ('arm_left_shoulder_pan.pos', -0.1),
    # Shoulder Lift
    'c': ('arm_left_shoulder_lift.pos', 0.1),
    'v': ('arm_left_shoulder_lift.pos', -0.1),
    # Elbow
    'b': ('arm_left_elbow_flex.pos', 0.1),
    'n': ('arm_left_elbow_flex.pos', -0.1),
    # Gripper
    'm': ('arm_left_gripper.pos', 0.1),
    ',': ('arm_left_gripper.pos', -0.1),
}

STOP_KEYS = [' ', 'k']

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def limit(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def zmq_listener(ip, port, stop_event):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(f"tcp://{ip}:{port}")
    socket.setsockopt(zmq.CONFLATE, 1)

    print(f"Listening for observations on {ip}:{port}...")

    while not stop_event.is_set():
        try:
            if socket.poll(timeout=100):
                msg = socket.recv_string()
                data = json.loads(msg)
                
                obs_decoded = {}
                for k, v in data.items():
                    # Heuristic to detect base64 encoded images
                    if isinstance(v, str) and len(v) > 1000:
                        try:
                            img_bytes = base64.b64decode(v)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is not None:
                                # Convert BGR to RGB for Rerun
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                # Prefix with "camera/" to match expected Rerun hierarchy if not already
                                if not k.startswith("camera/"):
                                    # Rename keys like 'head_top' -> 'camera/head_top'
                                    obs_decoded[f"camera/{k}"] = img
                                else:
                                    obs_decoded[k] = img
                        except Exception:
                            obs_decoded[k] = v
                    else:
                        obs_decoded[k] = v
                
                log_rerun_data(observation=obs_decoded)
                
        except Exception as e:
            # print(f"Error in ZMQ listener: {e}")
            pass

    socket.close()
    context.term()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP, help="IP address of the robot/simulation")
    args = parser.parse_args()
    
    global settings
    settings = termios.tcgetattr(sys.stdin)
    
    # Init Rerun
    init_rerun(session_name="alohamini_remote_teleop")
    
    # Start ZMQ Listener Thread
    stop_event = threading.Event()
    listener_thread = threading.Thread(target=zmq_listener, args=(args.ip, OBS_PORT, stop_event), daemon=True)
    listener_thread.start()
    
    # Command Socket
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{args.ip}:{CMD_PORT}")
    
    print(f"Connected to command port {CMD_PORT} at {args.ip}")

    # State
    target_state = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "lift_axis.height_mm": 0.0,
        # Initialize arm joints to 0
        "arm_left_shoulder_pan.pos": 0.0,
        "arm_left_shoulder_lift.pos": 0.0,
        "arm_left_elbow_flex.pos": 0.0,
        "arm_left_wrist_flex.pos": 0.0,
        "arm_left_wrist_roll.pos": 0.0,
        "arm_left_gripper.pos": 0.0,
    }
    
    is_recording = False
    
    try:
        print(msg)
        while True:
            key = getKey()
            
            if key == '\x03': # CTRL-C
                break
                
            # Recording Toggle
            if key == 'p': # Changed from 'r' to 'p' for record to free up 'r' for roll
                is_recording = not is_recording
                cmd_key = "start_recording" if is_recording else "stop_recording"
                cmd_socket.send_string(json.dumps({cmd_key: True}))
                print(f"Recording: {is_recording}")
                rr.log("is_recording", rr.Scalars(1.0 if is_recording else 0.0))
                continue

            # Reset
            if key == '0':
                print("Sending RESET command...")
                cmd_socket.send_string(json.dumps({"reset": True}))
                # Reset local state too
                target_state = {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "lift_axis.height_mm": 0.0,
                    "arm_left_shoulder_pan.pos": 0.0,
                    "arm_left_shoulder_lift.pos": 0.0,
                    "arm_left_elbow_flex.pos": 0.0,
                    "arm_left_wrist_flex.pos": 0.0,
                    "arm_left_wrist_roll.pos": 0.0,
                    "arm_left_gripper.pos": 0.0,
                }
                continue

            # Movement
            if key in MOVE_BINDINGS:
                attr, val = MOVE_BINDINGS[key]
                target_state[attr] += val
                print(f"Cmd: {attr} = {target_state[attr]:.2f}")
                
            elif key in LIFT_BINDINGS:
                attr, val = LIFT_BINDINGS[key]
                target_state[attr] += val
                print(f"Lift: {target_state[attr]:.2f}")

            elif key in ARM_BINDINGS:
                attr, val = ARM_BINDINGS[key]
                target_state[attr] += val
                print(f"Arm: {attr} = {target_state[attr]:.2f}")
                
            elif key in STOP_KEYS:
                target_state["x.vel"] = 0.0
                target_state["y.vel"] = 0.0
                target_state["theta.vel"] = 0.0
                print("STOPPED")

            # Limits
            target_state["x.vel"] = limit(target_state["x.vel"], -0.5, 0.5)
            target_state["y.vel"] = limit(target_state["y.vel"], -0.5, 0.5)
            # target_state["theta.vel"] = limit(target_state["theta.vel"], -5.0, 5.0)
            
            # Clamp lift to valid range (0 to 600mm)
            target_state["lift_axis.height_mm"] = limit(target_state["lift_axis.height_mm"], 0.0, 600.0)

            # Send Command
            cmd_socket.send_string(json.dumps(target_state))
            
            # Log Action to Rerun
            log_rerun_data(action=target_state)

    except Exception as e:
        print(e)
        
    finally:
        stop_event.set()
        # Stop robot
        final_stop = {k: 0.0 for k in target_state}
        final_stop["lift_axis.height_mm"] = target_state["lift_axis.height_mm"]
        cmd_socket.send_string(json.dumps(final_stop))
        
        cmd_socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == "__main__":
    main()
