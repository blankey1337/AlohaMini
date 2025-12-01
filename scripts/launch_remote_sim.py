#!/usr/bin/env python3
import subprocess
import sys
import time
import threading
import os
import signal
import re

# Configuration
PEM_FILE = "AlohaMini.pem"
REMOTE_HOST = "ubuntu@64.181.241.126"
REMOTE_IP = "64.181.241.126"

# Commands
CMD_SIM_DOCKER = [
    "ssh", "-i", PEM_FILE, REMOTE_HOST, "-t",
    'sudo docker run --name isaac-sim --privileged --entrypoint bash -it --gpus all '
    '-e "ACCEPT_EULA=Y" -e "NVIDIA_VISIBLE_DEVICES=all" -e "NVIDIA_DRIVER_CAPABILITIES=all" '
    '--rm --network=host -v ~/AlohaMini:/isaac-sim/AlohaMini nvcr.io/nvidia/isaac-sim:4.2.0 '
    '-c "cd /isaac-sim && ./python.sh -m pip install \\"numpy<2.0.0\\" opencv-python-headless pyzmq && '
    './python.sh AlohaMini/software/examples/alohamini/isaac_sim/isaac_alohamini_env.py '
    '--/app/livestream/enabled=true --/app/livestream/proto=ws"'
]

CMD_PORT_FWD = [
    "ssh", "-i", PEM_FILE, "-N",
    "-L", "5555:localhost:5555",
    "-L", "5556:localhost:5556",
    "-L", "8211:localhost:8211",
    REMOTE_HOST
]

CMD_LOCAL_TELEOP = [
    sys.executable, "software/examples/alohamini/remote_teleop_rerun.py"
]

def stream_reader(pipe, prefix, stop_event, on_match=None, match_pattern=None):
    """Reads from a pipe and prints to stdout, optionally triggering a callback on match."""
    try:
        for line in iter(pipe.readline, ''):
            if stop_event.is_set():
                break
            if not line:
                break
            # Print output with prefix
            # Clean up the line: handle \r and whitespace
            # Use \r\n to ensure we move to the next line properly
            clean_line = line.replace('\r', '').strip()
            
            if clean_line:
                print(f"[{prefix}] {clean_line}\r")
            
            if match_pattern and on_match:
                if match_pattern in line:
                    on_match()
                    # We only trigger once
                    on_match = None 
    except Exception:
        pass

def cleanup_remote():
    """Stops the remote docker container and ensures no lingering processes."""
    print("Cleaning up remote environment...")
    cmd = ["ssh", "-i", PEM_FILE, REMOTE_HOST, "sudo docker stop isaac-sim || true && sudo docker rm isaac-sim || true"]
    try:
        subprocess.run(cmd, check=False, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Warning: Remote cleanup failed: {e}")

def sync_code():
    print("Syncing local code to remote server...")
    # Sync software folder to ensure remote has latest changes
    # This creates ~/AlohaMini/software if it doesn't exist, or updates it
    cmd = ["scp", "-i", PEM_FILE, "-r", "software", f"{REMOTE_HOST}:~/AlohaMini/"]
    try:
        subprocess.run(cmd, check=True)
        print("Sync complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing code: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(PEM_FILE):
        print(f"Error: {PEM_FILE} not found in current directory.")
        print("Please run this script from the project root where the .pem file is located.")
        sys.exit(1)

    print("Starting Remote Isaac Sim Workflow...")
    print("-------------------------------------")

    # Initial cleanup to ensure fresh state
    cleanup_remote()

    # Sync code
    sync_code()

    # Events and State
    stop_event = threading.Event()
    sim_ready_event = threading.Event()
    
    processes = []

    def on_sim_ready():
        print("\n>>> SIMULATION READY DETECTED! Starting Port Forwarding... <<<\n")
        sim_ready_event.set()

    # 1. Start Simulation (Remote Docker)
    print(f"Step 1: Launching Isaac Sim container on {REMOTE_HOST}...")
    sim_process = subprocess.Popen(
        CMD_SIM_DOCKER,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    processes.append(sim_process)

    # Start monitoring thread for Sim output
    sim_thread = threading.Thread(
        target=stream_reader, 
        args=(sim_process.stdout, "REMOTE", stop_event, on_sim_ready, "Isaac Sim AlohaMini running")
    )
    sim_thread.daemon = True
    sim_thread.start()

    try:
        # Wait for Sim to be ready
        print("Waiting for simulation to initialize (this may take a minute)...")
        while not sim_ready_event.is_set():
            if sim_process.poll() is not None:
                print("Error: Simulation process exited prematurely.")
                sys.exit(1)
            time.sleep(1)

        # 2. Start Port Forwarding
        print("Step 2: Starting SSH Port Forwarding...")
        fwd_process = subprocess.Popen(
            CMD_PORT_FWD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(fwd_process)
        
        # Give it a moment to establish
        time.sleep(3)
        if fwd_process.poll() is not None:
             print("Error: Port forwarding failed to start.")
             # Check stderr
             print(fwd_process.stderr.read().decode())
             # Continue anyway? No, teleop needs it.
             sys.exit(1)
        else:
             print("Port forwarding established.")

        # 3. Start Local Teleop
        print("Step 3: Starting Local Teleop...")
        print("-------------------------------------")
        print("Use the keyboard commands below to control the robot.")
        
        teleop_process = subprocess.Popen(CMD_LOCAL_TELEOP)
        processes.append(teleop_process)
        
        # Wait for teleop to finish (user quits)
        teleop_process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping workflow...")
    finally:
        stop_event.set()
        print("Cleaning up processes...")
        
        # Kill all started processes
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        
        # Explicit remote cleanup
        cleanup_remote()
                    
        print("Done.")

if __name__ == "__main__":
    main()
