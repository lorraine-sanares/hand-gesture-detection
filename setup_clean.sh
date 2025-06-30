#!/bin/bash
cd ~/Documents/Github/hand-gesture-detection

# Clean up
rm -rf venv
pip3 uninstall mediapipe mediapipe-rpi4 -y

# Remove broken system version (if it exists)
rm -rf ~/.local/lib/python3.9/site-packages/mediapipe*

# Set up venv
python3 -m venv venv
source venv/bin/activate

# Reinstall
pip install --upgrade pip
pip install opencv-python "numpy<2.0"
pip install mediapipe-rpi4 --prefer-binary -i https://www.piwheels.org/simple

# Test it
python -c "import mediapipe as mp; print(mp.__file__)"
