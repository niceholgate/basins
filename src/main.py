import src.basins as basins
import src.types as types
import src.utils as utils

import numpy as np
import uuid

from pydantic import ValidationError
from fastapi import FastAPI, BackgroundTasks, Response, status
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    # "localhost:3000",
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
def load_rgb_frame(uuid: str, frame: int, response: Response):
    try:
        rgb_frame_data_path = utils.get_images_dir(uuid) / utils.get_frame_filename(frame, 'txt')
        return np.genfromtxt(rgb_frame_data_path, dtype=np.int_).tolist()
    except Exception as err:
        print(err)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str(err)}
