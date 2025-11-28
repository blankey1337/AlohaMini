# AlohaMini Isaac Sim Integration

This directory contains tools to simulate AlohaMini using NVIDIA Isaac Sim.

## Prerequisites

1.  **NVIDIA Isaac Sim**: You must have NVIDIA Isaac Sim installed on a machine with a supported NVIDIA GPU.
    *   This can be a local machine or a remote server (cloud instance).
2.  **Environment**: Run these scripts using the python environment provided by Isaac Sim (usually `./python.sh` in the Isaac Sim directory).

## Usage

### 1. Start the Simulation (Remote or Local)

On the machine with the GPU (Server):

```bash
# From the root of the repo, assuming `isaac_sim_python` is alias to Isaac Sim's python
isaac_sim_python software/examples/alohamini/isaac_sim/isaac_alohamini_env.py
```

*   The simulation listens on port **5555** (Commands) and publishes on port **5556** (Observations).
*   Ensure these ports are open if connecting directly.

### 2. Teleoperation (Client)

You can run the teleoperation script on your local machine (e.g., MacBook) to control the robot remotely.

#### Option A: Direct Connection (VPN/LAN)

If you can ping the server IP directly:

```bash
# Replace 192.168.1.100 with your server's IP
python software/examples/alohamini/standalone_teleop.py --ip 192.168.1.100
```

#### Option B: SSH Tunneling (Recommended for Cloud)

If your server is behind a firewall or on the cloud, use SSH tunneling to forward the ports.

1.  **Establish Tunnel**:
    ```bash
    # Forward local ports 5555/5556 to remote localhost:5555/5556
    ssh -L 5555:localhost:5555 -L 5556:localhost:5556 user@remote-server-ip
    ```

2.  **Run Teleop locally**:
    ```bash
    # Connect to localhost (which is tunneled to remote)
    python software/examples/alohamini/standalone_teleop.py --ip 127.0.0.1
    ```

### 3. Verify Connection

You can use the verify script to check if observations are being received.

```bash
python software/examples/debug/verify_sim_output.py --ip <SERVER_IP>
```
