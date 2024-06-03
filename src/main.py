import src.basins as basins
import src.types as types
import src.imaging as imaging
import src.utils as utils

import uuid

from pydantic import ValidationError
from fastapi import FastAPI, BackgroundTasks, Response, status, Header
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

app = FastAPI()

origins = [
    "http://localhost:3000",
    'http://192.168.0.106:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post('/create/still', status_code=202)
async def create_still(request: types.StillRequest, response: Response, background_tasks: BackgroundTasks):
    try:
        params = types.StillParameters.from_request(request)
        this_uuid = str(uuid.uuid4())
        background_tasks.add_task(
            basins.create_still,
            this_uuid, params)
        return {'message': 'Creating an image',
                'id': this_uuid}
    except ValidationError as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str({error['loc'][0]: error['msg'] for error in e.errors()})}


@app.post('/create/animation', status_code=202)
async def create_animation(request: types.AnimationRequest, response: Response, background_tasks: BackgroundTasks):
    try:
        params = types.AnimationParameters.from_request(request)
        this_uuid = str(uuid.uuid4())
        background_tasks.add_task(
            basins.create_animation,
            this_uuid, params)
        return {'message': 'Creating an animation',
                'id': this_uuid}
    except ValidationError as err:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str({error['loc'][0]: error['msg'] for error in err.errors()})}
    except ValueError as err:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str(err)}
    except Exception as err:
        print(err)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str(err)}


@app.get('/load/rgb_frame/{uuid}/{frame}')
def get_rgb_frame(uuid: str, frame: int, response: Response):
    try:
        imaging.load_rgb_file(utils.get_images_dir(uuid), frame).tolist()
    except Exception as err:
        print(err)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str(err)}



# TODO: ability to look at a selection of frames in the video (while they are being produced),
# then when it's finished to click a button to generate and download a file.
@app.get("/video")
async def video_endpoint(range: str = Header(None)):
    CHUNK_SIZE = 1024 * 1024
    video_path = Path("C:/dev/python_projects/basins/images/video3.webm")
    start, end = range.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else start + CHUNK_SIZE
    with open(video_path, "rb") as video:
        video.seek(start)
        data = video.read(end - start)
        filesize = str(video_path.stat().st_size)
        headers = {
            'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
            'Accept-Ranges': 'bytes'
        }
        return Response(data, status_code=206, headers=headers, media_type="video/mp4")