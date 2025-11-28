# Remote Simulation & Training Workflow

This guide outlines how to develop, train, and test AlohaMini using a remote cloud GPU (e.g., Lambda Labs, AWS) for the heavy simulation, while controlling everything from your local laptop (e.g., MacBook).

## Architecture

*   **Cloud Server (The "Lab")**: Runs NVIDIA Isaac Sim. Handles physics, rendering, and training.
*   **Local Machine (The "Mission Control")**: Runs the Dashboard and Teleoperation scripts. Connects to the cloud via SSH.

## Prerequisites

1.  **Cloud Instance**: A server with an NVIDIA RTX GPU (A10, A100, RTX 3090/4090).
    *   Recommended: Lambda Labs or Brev.dev (Ubuntu 20.04/22.04).
    *   Must have **NVIDIA Drivers** and **Isaac Sim** installed (or use the Isaac Sim Docker container).
2.  **Local Machine**: Your laptop (Mac/Windows/Linux).
3.  **SSH Access**: You must be able to SSH into the cloud instance.

## Setup

### 1. Cloud Server Setup
1.  SSH into your cloud instance.
2.  Clone this repository:
    ```bash
    git clone https://github.com/blankey1337/AlohaMini.git
    cd AlohaMini
    ```
3.  Ensure you are in the python environment that has access to Isaac Sim (often `./python.sh` in the Isaac Sim folder).

### 2. Local Machine Setup
1.  Clone this repository locally.
2.  Install dependencies:
    ```bash
    pip install -r software/requirements.txt
    ```

## The Workflow

### Phase 1: Data Collection

1.  **Start the Simulation (Cloud)**
    Run the simulation environment script. This listens on ports 5555 (Cmd) and 5556 (Obs).
    ```bash
    # On Cloud
    isaac_sim_python software/examples/alohamini/isaac_sim/isaac_alohamini_env.py
    ```

2.  **Establish Connection (Local)**
    Forward the ZMQ ports from the cloud to your localhost.
    ```bash
    # On Local Mac
    ssh -L 5555:localhost:5555 -L 5556:localhost:5556 ubuntu@<CLOUD_IP>
    ```

3.  **Launch Dashboard (Local)**
    Start the web dashboard to see what the robot sees.
    ```bash
    # On Local Mac
    python software/dashboard/app.py
    ```
    Open `http://localhost:5001` in your browser.

4.  **Teleoperate & Record**
    *   Use the Dashboard to see the camera feed.
    *   Run the teleop script in another terminal to control the robot with your keyboard:
        ```bash
        python software/examples/alohamini/standalone_teleop.py --ip 127.0.0.1
        ```
    *   **To Record**: Click the **"Start Recording"** button on the Dashboard.
    *   Perform the task (e.g., pick up the object).
    *   Click **"Stop Recording"**.
    *   Repeat 50-100 times. The data is saved to `AlohaMini/data_sim/` on the **Cloud Server**.

### Phase 2: Training

You train the model directly on the Cloud GPU where the data lives.

1.  **Stop the Simulation** (to free up GPU VRAM).
2.  **Run Training**:
    Use the LeRobot training script (or your custom training script) pointing to the generated dataset.
    ```bash
    # On Cloud
    python software/src/lerobot/scripts/train.py \
        --dataset data_sim \
        --policy act \
        --batch_size 8 \
        --num_epochs 1000
    ```
    *Note: Exact training command depends on the LeRobot configuration.*

3.  **Output**: This produces a model file (e.g., `outputs/train/policy.safetensors`).

### Phase 3: Evaluation

Test the trained model in the simulator to see if it works.

1.  **Restart Simulation (Cloud)**:
    ```bash
    isaac_sim_python software/examples/alohamini/isaac_sim/isaac_alohamini_env.py
    ```
2.  **Run Inference Node (Cloud or Local)**:
    You need a script that loads the model and closes the loop (reads obs -> runs model -> sends action).
    *   *Coming Soon: `eval_sim.py` which loads the safetensor and drives the ZMQ robot.*

3.  **Watch (Local)**:
    Use the Dashboard to watch the robot perform the task autonomously.

## Troubleshooting

*   **Laggy Video**: ZMQ over SSH tunneling is usually fast enough for 640x480, but if it lags, check your internet connection speed to the cloud server.
*   **"Address already in use"**: Ensure no other python scripts are using ports 5555/5556 on either machine.
