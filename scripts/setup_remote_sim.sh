#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/blankey1337/AlohaMini.git"
ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:2023.1.1"

echo ">>> Starting Remote Setup..."

# 1. Install Docker & NVIDIA Container Toolkit if missing
if ! command -v docker > /dev/null 2>&1; then
    echo ">>> Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker "$USER"
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo ">>> Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# 2. Docker Login
if [ -z "$NVIDIA_API_KEY" ]; then
    echo ">>> Please export NVIDIA_API_KEY before running this script."
    exit 1
fi

echo ">>> Logging into NGC..."
echo "$NVIDIA_API_KEY" | sudo docker login nvcr.io -u '$oauthtoken' --password-stdin

# 3. Clone Repo
if [ -d "AlohaMini" ]; then
    if [ -d "AlohaMini/.git" ]; then
        echo ">>> AlohaMini already cloned. Pulling latest..."
        cd AlohaMini && git pull && cd ..
    else
        echo ">>> AlohaMini directory exists but is not a git repo. Removing and cloning..."
        rm -rf AlohaMini
        git clone "$REPO_URL"
    fi
else
    echo ">>> Cloning AlohaMini..."
    git clone "$REPO_URL"
fi

# 4. Pull Isaac Sim Image
echo ">>> Pulling Isaac Sim Image ($ISAAC_IMAGE)..."
sudo docker pull "$ISAAC_IMAGE"

echo ">>> Setup Complete!"
