import json
import threading
import time
import base64
import zmq
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Global state
latest_observation = {}
lock = threading.Lock()
connected = False

def zmq_worker(ip='127.0.0.1', port=5556):
    global latest_observation, connected
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(f"tcp://{ip}:{port}")
    socket.setsockopt(zmq.CONFLATE, 1)
    
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
        frame_data = None
        with lock:
            if camera_name in latest_observation:
                b64_str = latest_observation[camera_name]
                if b64_str:
                    try:
                        # It is already a base64 encoded JPG from the host
                        frame_data = base64.b64decode(b64_str)
                    except Exception:
                        pass
        
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
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

@app.route('/api/status')
def get_status():
    with lock:
        # Filter out large image data for status endpoint
        status = {k: v for k, v in latest_observation.items() if not (isinstance(v, str) and len(v) > 1000)}
        status['connected'] = connected
    return jsonify(status)

if __name__ == '__main__':
    # Start ZMQ thread
    t = threading.Thread(target=zmq_worker, daemon=True)
    t.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
