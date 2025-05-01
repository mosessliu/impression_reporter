# Cheat Sheet

## Linting
Install ruff:  
`pip install ruff`

Run ruff:  
`ruff check . --fix`

## Docker
Build the image:  
`sudo docker build -t impression-reporter .`  

Run the image:  
`sudo docker run -it --env-file .env impression-reporter`

Run the image in Jetson:  
`sudo docker run -it --env-file .env --rm --ipc host --network host --shm-size 14G --runtime nvidia --device /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY impression-reporter`

Run the app in Jetson:  
`cd src`  
`python3 main.py --camera 0 /opt/nanoowl/data/owl_image_encoder_patch32.engine`
