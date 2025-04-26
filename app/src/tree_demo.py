import asyncio
import argparse
from aiohttp import web, WSCloseCode, ClientSession
import logging
import weakref
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from nanoowl.tree import Tree
from nanoowl.tree_predictor import (
    TreePredictor
)
from nanoowl.tree_drawing import draw_tree_output
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
    
    REPORTING_URL = 'https://eovs5rjtnt5cyrs.m.pipedream.net'

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )

    prompt_data = None
    detection_cache = []

    def get_colors(count: int):
        cmap = plt.cm.get_cmap("rainbow", count)
        colors = []
        for i in range(count):
            color = cmap(i)
            color = [int(255 * value) for value in color]
            colors.append(tuple(color))
        return colors


    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)


    async def handle_index_get(request: web.Request):
        logging.info("handle_index_get")
        return web.FileResponse("./index.html")


    async def websocket_handler(request):

        global prompt_data

        ws = web.WebSocketResponse()

        await ws.prepare(request)

        logging.info("Websocket connected.")

        request.app['websockets'].add(ws)

        try:
            async for msg in ws:
                logging.info("Received message from websocket.")
                if "prompt" in msg.data:
                    header, prompt = msg.data.split(":")
                    logging.info("Received prompt: " + prompt)
                    try:
                        tree = Tree.from_prompt(prompt)
                        clip_encodings = predictor.encode_clip_text(tree)
                        owl_encodings = predictor.encode_owl_text(tree)
                        prompt_data = {
                            "tree": tree,
                            "clip_encodings": clip_encodings,
                            "owl_encodings": owl_encodings
                        }
                        logging.info("Set prompt: " + prompt)
                    except Exception as e:
                        print(e)
        finally:
            request.app['websockets'].discard(ws)

        return ws


    async def on_shutdown(app: web.Application):
        for ws in set(app['websockets']):
            await ws.close(code=WSCloseCode.GOING_AWAY,
                        message='Server shutdown')
            
    def log_detections(detections):
        detected = bool(list(detections.detections)[1:])
        detection_cache.append(detected)

    async def report_detections():
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
                        print("Something went wrong.")
                        
            detection_cache.clear()
                

    async def detection_loop(app: web.Application):

        loop = asyncio.get_running_loop()

        logging.info("Opening camera.")

        camera = cv2.VideoCapture(CAMERA_DEVICE)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        logging.info("Loading predictor.")

        def _read_and_encode_image():

            re, image = camera.read()

            if not re:
                return re, None

            image_pil = cv2_to_pil(image)

            if prompt_data is not None:
                prompt_data_local = prompt_data
                detections = predictor.predict(
                    image_pil,
                    tree=prompt_data_local['tree'],
                    clip_text_encodings=prompt_data_local['clip_encodings'],
                    owl_text_encodings=prompt_data_local['owl_encodings']
                )
                log_detections(detections)
                image = draw_tree_output(image, detections, prompt_data_local['tree'])

            image_jpeg = bytes(
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
            )

            return re, image_jpeg

        while True:

            re, image = await loop.run_in_executor(None, _read_and_encode_image)
            
            if not re:
                break
            
            for ws in app["websockets"]:
                await ws.send_bytes(image)

        camera.release()


    async def run_detection_loop(app):
        try:
            task = asyncio.create_task(detection_loop(app))
            yield
            task.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            await task

    async def run_report_detections(_):
        try:
            task = asyncio.create_task(report_detections())
            yield
            task.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            await task

    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)
    app.cleanup_ctx.append(run_report_detections)
    web.run_app(app, host=args.host, port=args.port)