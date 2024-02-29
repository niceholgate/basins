import src.basins as basins


import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
app = FastAPI()


class StillRequest(BaseModel):
    colour_set: int = 1


class AnimationRequest(BaseModel):
    colour_set: int = 1
    delta: int
    frames: int
    fps: int


@app.post('/create/still', status_code=202)
async def create_still(request: StillRequest, background_tasks: BackgroundTasks):
    this_uuid = str(uuid.uuid4())
    background_tasks.add_task(
        basins.create_still,
        this_uuid, request.colour_set)
    return {'message': 'Creating an image',
            'id': this_uuid}


@app.post('/create/animation', status_code=202)
async def create_animation(request: AnimationRequest, background_tasks: BackgroundTasks):
    this_uuid = str(uuid.uuid4())
    background_tasks.add_task(
        basins.create_animation,
        this_uuid, request.colour_set, request.delta, request.frames, request.fps)
    return {'message': 'Creating an animation',
            'id': this_uuid}
