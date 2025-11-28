import time
import json
import zmq

# --- Config ---
CMD_PORT = 5555
IP = "127.0.0.1"

def main():
    context = zmq.Context()
    print(f"Connecting to command port {CMD_PORT}...")
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{IP}:{CMD_PORT}")

    print("Get ready! Switching to dashboard in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Starting automated dance sequence (Looping 3 times)...")
    
    # Define a sequence of actions: (duration_s, action_dict)
    sequence = [
        (2.0, {"x.vel": 0.2, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}),   # Forward
        (2.0, {"x.vel": -0.2, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}),  # Backward
        (2.0, {"x.vel": 0.0, "y.vel": 0.2, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}),   # Slide Left
        (2.0, {"x.vel": 0.0, "y.vel": -0.2, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}),  # Slide Right
        (2.0, {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 60.0, "lift_axis.height_mm": 50.0}), # Rotate + Lift Up
        (2.0, {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -60.0, "lift_axis.height_mm": 0.0}), # Rotate Back + Lift Down
        (1.0, {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}),   # Stop
    ]

    try:
        for i in range(3):
            print(f"--- Loop {i+1}/3 ---")
            for duration, action in sequence:
                print(f"Executing: {action}")
                start = time.time()
                while time.time() - start < duration:
                    cmd_socket.send_string(json.dumps(action))
                    time.sleep(0.05) # Send at 20Hz
        
        print("Sequence complete.")

    except KeyboardInterrupt:
        pass
    finally:
        # Stop
        stop_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}
        cmd_socket.send_string(json.dumps(stop_action))
        cmd_socket.close()
        context.term()

if __name__ == "__main__":
    main()
