import argparse
import json
import select
import sys
import termios
import time
import tty

import zmq

# --- Config ---
CMD_PORT = 5555
DEFAULT_IP = "127.0.0.1"

msg = """
Reading from the keyboard  and Publishing to ZMQ!
---------------------------
Moving around:
   q    w    e
   a    s    d
   z    x    c

w/x : increase/decrease linear x speed (Forward/Backward)
a/d : increase/decrease linear y speed (Left/Right)
q/e : increase/decrease angular speed (Rotate Left/Right)

u/j : increase/decrease lift height

space key, k : force stop
CTRL-C to quit
"""

# Key mappings for velocity updates
# (key) : (attribute, increment_value)
MOVE_BINDINGS = {
    'w': ('x.vel', 0.05),
    's': ('x.vel', -0.05),
    'a': ('y.vel', 0.05),
    'd': ('y.vel', -0.05),
    'q': ('theta.vel', 10.0),
    'e': ('theta.vel', -10.0),
}

# Key mappings for lift (positional)
LIFT_BINDINGS = {
    'u': ('lift_axis.height_mm', 2.0),
    'j': ('lift_axis.height_mm', -2.0),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP, help="IP address of the robot/simulation")
    args = parser.parse_args()
    
    settings = termios.tcgetattr(sys.stdin)
    
    # ZMQ Setup
    context = zmq.Context()
    print(f"Connecting to command port {CMD_PORT} at {args.ip}...")
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{args.ip}:{CMD_PORT}")

    # State
    status = 0
    target_state = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "lift_axis.height_mm": 0.0
    }

    try:
        print(msg)
        while True:
            key = getKey()
            
            # Quit
            if key == '\x03': # CTRL-C
                break

            # Update State
            if key in MOVE_BINDINGS.keys():
                attr, val = MOVE_BINDINGS[key]
                target_state[attr] += val
                
                # Print status occasionally
                status = (status + 1) % 10
                if status == 0:
                    print(f"State: {target_state}")

            elif key in LIFT_BINDINGS.keys():
                attr, val = LIFT_BINDINGS[key]
                target_state[attr] += val
                print(f"Lift: {target_state[attr]}")

            elif key in STOP_KEYS:
                target_state["x.vel"] = 0.0
                target_state["y.vel"] = 0.0
                target_state["theta.vel"] = 0.0
                print("STOPPED")

            # Limits (Optional, to keep it sane)
            target_state["x.vel"] = limit(target_state["x.vel"], -0.5, 0.5)
            target_state["y.vel"] = limit(target_state["y.vel"], -0.5, 0.5)
            # target_state["theta.vel"] = limit(target_state["theta.vel"], -90, 90) # No limit on rotation usually needed but safer

            # Send
            cmd_socket.send_string(json.dumps(target_state))
            
            # Small sleep to prevent busy loop if we wanted, but select handles timing
            # time.sleep(0.01)

    except Exception as e:
        print(e)

    finally:
        # Stop robot on exit
        final_stop = {k: 0.0 for k in target_state}
        final_stop["lift_axis.height_mm"] = target_state["lift_axis.height_mm"] # Keep lift height
        cmd_socket.send_string(json.dumps(final_stop))
        
        cmd_socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
