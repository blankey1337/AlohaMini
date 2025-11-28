import argparse
import zmq
import cv2
import json
import base64
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="IP of the simulation/robot")
    parser.add_argument("--port", type=int, default=5556, help="Observation ZMQ port")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(f"tcp://{args.ip}:{args.port}")
    socket.setsockopt(zmq.CONFLATE, 1)

    print(f"Listening for video on {args.ip}:{args.port}...")
    print("Press 'q' to quit.")

    try:
        while True:
            # Non-blocking check
            if socket.poll(100):
                msg = socket.recv_string()
                data = json.loads(msg)
                
                # Look for images
                images = []
                for k, v in data.items():
                    # Heuristic: if key contains 'cam' or 'head' and value is string (base64)
                    if isinstance(v, str) and len(v) > 1000: 
                        try:
                            # Decode
                            jpg_original = base64.b64decode(v)
                            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                            img = cv2.imdecode(jpg_as_np, flags=1)
                            if img is not None:
                                # Add label
                                cv2.putText(img, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                images.append(img)
                        except:
                            pass
                            
                if images:
                    # Stack images horizontally
                    # Resize to same height if needed, but for now assume same size
                    h_min = min(img.shape[0] for img in images)
                    images_resized = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min)) for img in images]
                    
                    combined = np.hstack(images_resized)
                    cv2.imshow("Remote Stream", combined)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
