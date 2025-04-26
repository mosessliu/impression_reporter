import asyncio
import argparse
from aiohttp import ClientSession
import logging
import cv2
import PIL.Image
from nanoowl.tree import Tree
from nanoowl.tree_predictor import (
    TreePredictor
)
from nanoowl.owl_predictor import OwlPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--resolution", type=str, default="640x480", help="Camera resolution as WIDTHxHEIGHT")
    args = parser.parse_args()
    width, height = map(int, args.resolution.split("x"))

    CAMERA_DEVICE = args.camera
    IMAGE_QUALITY = args.image_quality
    PROMPT = "[a face [eyes]]"
    REPORTING_URL = 'https://eovs5rjtnt5cyrs.m.pipedream.net'

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )
    
    tree = Tree.from_prompt(PROMPT)
    clip_encodings = predictor.encode_clip_text(tree)
    owl_encodings = predictor.encode_owl_text(tree)
    prompt_data = {
        "tree": tree,
        "clip_encodings": clip_encodings,
        "owl_encodings": owl_encodings
    }
    
    detection_cache = []
    
    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)
    
    def log_detections(detections):
        detected = bool(list(detections.detections)[1:])
        detection_cache.append(detected)
    
    async def detection_loop():
        loop = asyncio.get_running_loop()

        print("Opening camera.")

        camera = cv2.VideoCapture(CAMERA_DEVICE)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print("Loading predictor.")

        def _read_and_encode_image():
            re, image = camera.read()

            if not re:
                return re

            image_pil = cv2_to_pil(image)

            prompt_data_local = prompt_data
            detections = predictor.predict(
                image_pil,
                tree=prompt_data_local['tree'],
                clip_text_encodings=prompt_data_local['clip_encodings'],
                owl_text_encodings=prompt_data_local['owl_encodings']
            )
            log_detections(detections)
                
            return re

        while True:

            re = await loop.run_in_executor(None, _read_and_encode_image)
            
            if not re:
                break

        camera.release()
        
    async def reporting_loop():
        while True:
            await asyncio.sleep(5)
            
            detections = [flag for flag in detection_cache if flag]
            data = {
                "detection_count": len(detections),
                "total_count": len(detection_cache)
            }
            print(data)
            
            async with ClientSession() as session:
                async with session.post(REPORTING_URL, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(result)
                    else:
                        logging.error("Something went wrong.")
                        
            detection_cache.clear()
            
    async def main():
        await asyncio.gather(
            asyncio.create_task(detection_loop()),
            asyncio.create_task(reporting_loop())
        )
        
    asyncio.run(main())
