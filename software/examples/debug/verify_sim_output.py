import zmq
import json
import argparse

def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="IP address of the simulation")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(f"tcp://{args.ip}:5556")
    
    print("Waiting for observation...")
    try:
        # Wait for a message with timeout
        if socket.poll(5000):
            msg = socket.recv_string()
            data = json.loads(msg)
            
            print("Received observation!")
            keys = data.keys()
            print(f"Keys: {list(keys)}")
            
            if "detections" in data:
                print(f"Detections found: {data['detections']}")
                if "head_front" in data["detections"]:
                    print("PASS: detections['head_front'] exists")
                    return True
                else:
                    print("FAIL: detections['head_front'] missing")
            else:
                print("FAIL: 'detections' key missing")
        else:
            print("FAIL: No message received in 5s")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    verify()
