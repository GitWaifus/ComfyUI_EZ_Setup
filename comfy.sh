#!/bin/bash

# Define the installation directory
INSTALL_DIR="/workspace/ComfyUI"

# Check if the ComfyUI directory already exists
if [ -d "$INSTALL_DIR" ]; then
    echo "ComfyUI directory already exists. Please remove the existing directory or pull the latest changes if you wish to update."
    exit 1
else
    # Clone ComfyUI repository
    git clone https://github.com/comfyanonymous/ComfyUI.git $INSTALL_DIR || { echo 'Failed to clone ComfyUI repository.'; exit 1; }
fi

# Move to the ComfyUI directory
cd $INSTALL_DIR || exit

# Check for an existing virtual environment
if [ -d "venv" ]; then
    echo "A virtual environment already exists. Activating it."
else
    # Set up virtual environment
    python3 -m venv venv || { echo 'Failed to create a virtual environment.'; exit 1; }
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip || { echo 'Failed to upgrade pip.'; exit 1; }

# Install requirements
pip install -r requirements.txt || { echo 'Failed to install requirements.'; exit 1; }

# Ensure setup.sh is executable and exists
if [ -f "/workspace/ComfyUI_EZ_Setup/setup.sh" ]; then
    chmod +x /workspace/setup.sh

    # Execute setup.sh script
    /workspace/setup.sh || { echo 'setup.sh failed'; exit 1; }
else
    echo "setup.sh does not exist in /workspace."
    exit 1
fi
