#!/bin/bash
# This is an install script to setup systemd service

set -e

SERVICE_NAME=impression-reporter
CONTAINER_NAME="impression-reporter-container"
IMAGE_NAME=impression-reporter:latest
USER_NAME=$(whoami)
DISPLAY_NUMBER=${DISPLAY:-:0}

# --- Create systemd service ---
echo "Creating systemd service..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF

[Unit]
Description=Run Impression Reporter application after valid network connection and display
After=network-online.target graphical.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
ExecStart=/usr/bin/docker run --rm \
    --env-file .env \
    --ipc host \
    --network host \
    --shm-size 14G \
    --runtime nvidia \
    --device /dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY_NUMBER \
    --privileged \
    $IMAGE_NAME \
    bash -c "cd /app/src && python3 main.py --camera 0 /opt/nanoowl/data/owl_image_encoder_patch32.engine"
    
ExecStop=/usr/bin/docker stop $CONTAINER_NAME
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=graphical.target
EOF

# --- Reload systemd, enable and start service ---
echo "Reloading systemd and enabling service..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME.service"
sudo systemctl restart "$SERVICE_NAME.service"

echo "Setup complete. Service '$SERVICE_NAME' is running as '$USER_NAME'."