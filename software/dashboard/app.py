import json
import threading
import time
import base64
import zmq
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# Global state
latest_observation = {}
lock = threading.Lock()
connected = False
recording = False
cmd_socket = None

def zmq_worker(ip='127.0.0.1', port=5556, cmd_port=5555):
    global latest_observation, connected, cmd_socket
    context = zmq.Context()
    
    # Sub Socket
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(f"tcp://{ip}:{port}")
    socket.setsockopt(zmq.CONFLATE, 1)
    
    # Cmd Socket (Push to Sim)
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{ip}:{cmd_port}")
    
    print(f"Connecting to ZMQ Stream at {ip}:{port}...")
    
    while True:
        try:
            msg = socket.recv_string()
            data = json.loads(msg)
            
            with lock:
                latest_observation = data
                connected = True
        except Exception as e:
            print(f"Error in ZMQ worker: {e}")
            connected = False
            time.sleep(1)

def generate_frames(camera_name):
    while True:
        frame_bytes = None
        detections = []
        
        with lock:
            if camera_name in latest_observation:
                b64_str = latest_observation[camera_name]
                if b64_str:
                    try:
                        frame_bytes = base64.b64decode(b64_str)
                    except Exception:
                        pass
            
            # Get detections
            raw_dets = latest_observation.get("detections", {})
            if isinstance(raw_dets, dict):
                detections = raw_dets.get(camera_name, [])
        
        if frame_bytes:
            # Decode to image to draw on it
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Draw detections
                for det in detections:
                    box = det.get("box", [])
                    label = det.get("label", "obj")
                    if len(box) == 4:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Re-encode
                ret, buffer = cv2.imencode('.jpg', img)
                if ret:
                    frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Return a blank or placeholder image if no data
            pass
            
        time.sleep(0.05) # Limit FPS for browser

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    return Response(generate_frames(camera_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/command', methods=['POST'])
def send_command():
    global cmd_socket
    if not request.json or 'command' not in request.json:
        return jsonify({'error': 'No command provided'}), 400
    
    cmd = request.json['command']
    print(f"Received command: {cmd}")
    
    # Example handling
    if cmd == 'reset_sim':
        # Send reset command (Isaac Sim needs to handle this logic)
        # For now, we can just zero out velocities or send a special flag
        if cmd_socket:
            cmd_socket.send_string(json.dumps({"reset": True}))
            
    elif cmd == 'start_recording':
        # Trigger recording logic (would need to signal record_bi.py or similar)
        pass
        
    return jsonify({'status': 'ok'})

@app.route('/api/status')
def get_status():
    with lock:
        # Filter out large image data for status endpoint, but keep the key
        status = {}
        for k, v in latest_observation.items():
            if isinstance(v, str) and len(v) > 1000:
                status[k] = "__IMAGE_DATA__"
            else:
                status[k] = v
        status['connected'] = connected
    return jsonify(status)

if __name__ == '__main__':
    # Start ZMQ thread
    t = threading.Thread(target=zmq_worker, daemon=True)
    t.start()
    
    app.run(host='0.0.0.0', port=5001, debug=False)
