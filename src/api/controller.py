import src.api.requests as types
import src.imaging as imaging
import src.utils as utils
import src.solving.interface as solve_interface
import src.config as cfg

import uuid

import uvicorn
from datetime import datetime
from pydantic import ValidationError
from fastapi import FastAPI, BackgroundTasks, Response, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

import logging
logger = logging.getLogger(__name__)


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

# {
#     "x_pixels": 400,
#     "y_pixels": 400,
#     "expressions": ["x**2+2*(y-6)**2-10-sin(2*x+d)", "2**(-x**2)-y/6+1+sin(x+d)/10"],
#     "colour_set": 1,
#     "delta": 6.283185307,
#     "frames": 300,
#     "fps": 30
# } why does this only find 2 solutions? there are 4.
@app.post('/create/still', status_code=202)
async def create_still(request: types.StillRequest, response: Response, background_tasks: BackgroundTasks):
    try:
        params = types.StillParameters.from_request(request)
        this_uuid = str(uuid.uuid4())

        utils.logger_setup(logger, utils.get_images_dir(this_uuid), 'still')
        logger.debug(request.model_dump_json())

        background_tasks.add_task(
            create_still_inner,
            this_uuid, params)
        return {
            'message': 'Creating an image',
            'id': this_uuid}

    except ValidationError as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str({error['loc'][0]: error['msg'] for error in e.errors()})}


@app.post('/create/animation', status_code=202)
async def create_animation(request: types.AnimationRequest, response: Response, background_tasks: BackgroundTasks):
    try:
        params = types.AnimationParameters.from_request(request)
        this_uuid = str(uuid.uuid4())

        utils.logger_setup(logger, utils.get_images_dir(this_uuid), 'animation')
        logger.debug(request.model_dump_json())

        background_tasks.add_task(
            create_animation_inner,
            this_uuid, params)
        response.body = {
            'message': 'Creating an animation',
            'id': this_uuid}
        return response
    except ValidationError as err:
        response.status_code = status.HTTP_400_BAD_REQUEST
        response.body = {'message': 'Input errors: ' + str({error['loc'][0]: error['msg'] for error in err.errors()})}
        return response
    except ValueError as err:
        response.status_code = status.HTTP_400_BAD_REQUEST
        response.body = {'message': 'Input errors: ' + str(err)}
        return response
    except Exception as err:
        print(err)
        response.status_code = status.HTTP_400_BAD_REQUEST
        response.body = {'message': 'Input errors: ' + str(err)}
        return response


@app.get('/load/{uuid}/rgb_frame/{frame}')
def load_rgb_frame(uuid: str, frame: int, response: Response):
    try:
        directory = utils.get_images_dir(uuid)
        rgb_frame_data_path = directory / utils.get_frame_filename(frame, 'txt')
        response.body = imaging.image.load_rgb_file(rgb_frame_data_path).tolist()
        return response
    except Exception as err:
        print(err)
        response.status_code = status.HTTP_400_BAD_REQUEST
        response.body = {'message': 'Input errors: ' + str(err)}
        return response


# TODO: ability to look at a selection of frames in the video (while they are being produced),
# then when it's finished to click a button to generate and download a file.
@app.get('/load/{uuid}/run_data')
def load_run_data(uuid: str, response: Response):
    try:
        data = imaging.image.load_run_data(uuid)
        if data:
            return data
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': f'Input errors: No run data found for uuid={uuid}'}
    except Exception as err:
        print(err)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {'message': 'Input errors: ' + str(err)}



# @app.get("/video")
# async def video_endpoint(range: str = Header(None)):
#     CHUNK_SIZE = 1024 * 1024
#     video_path = Path("C:/dev/python_projects/basins/images/video3.webm")
#     start, end = range.replace("bytes=", "").split("-")
#     start = int(start)
#     end = int(end) if end else start + CHUNK_SIZE
#     with open(video_path, "rb") as video:
#         video.seek(start)
#         data = video.read(end - start)
#         filesize = str(video_path.stat().st_size)
#         headers = {
#             'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
#             'Accept-Ranges': 'bytes'
#         }
#         return Response(data, status_code=206, headers=headers, media_type="video/mp4")


@utils.timed
def produce_image_timed(f_lambda, j_lambda, delta, images_dir, colour_set, unique_solutions, x_coords, y_coords, i):
    if cfg.ENABLE_QUADTREES:
        solutions_grid, iterations_grid = solve_interface.solve_grid_quadtrees_wrapper(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    else:
        solutions_grid, iterations_grid = solve_interface.solve_grid_wrapper(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    print('starting save')
    n = datetime.now()
    imaging.image.save_still(images_dir, solutions_grid.to_numpy(), iterations_grid.to_numpy(), unique_solutions, colour_set=colour_set, frame=i)
    print((datetime.now() - n).total_seconds() * 1000)
    print('finishing save')


# TODO:
# -improve UI layout + add tabs (sidebar) + refactoring
# -loading bars with server-sent events
###https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events?fbclid=IwZXh0bgNhZW0CMTAAAR0Jc8W85IGtv2YlQdb2PT3QU8-d7vmNGV8_vW6rgQxOKPOdzwto9SDveRI_aem_wJvZf1ZyYehwhxYwiRISCg
###https://medium.com/codex/implementation-of-server-sent-events-and-eventsource-live-progress-indicator-using-react-and-723596f35225
# -download video once all frames computed
# -live view of frame as it is generated?
# -add more logging
# -Consolidate input validations in one place
# -Animations can pan/zoom the grid
# -Complex equations
# -queue and RL requests
# -setup lambda, APIGateway, static app hosting, LocalStack
# -numba 'interface' concept might be cleaner as a module e.g. solving.interface uses classes in quad_tree and solve


def create_animation_inner(uuid: str, params: types.AnimationParameters):
    utils.logger_setup(logger, utils.get_images_dir(uuid), 'animation')
    # logger.debug(params.model_dump_json(exclude={'f_lambda', 'j_lambda'}))

    images_dir = utils.get_images_dir(uuid)
    utils.mkdir_if_nonexistent(images_dir)

    unique_solutions = solve_interface.find_unique_solutions_wrapper(params.f_lambda, params.j_lambda, params.deltas[0], np.array(params.search_limits))
    if len(unique_solutions) < 2:
        raise Exception('There are no solutions!' if len(unique_solutions) == 0 else 'There is only 1 solution!')

    # Update this as new solutions are found - the ordering will be kept consistent so that each solution always is given the same colour in each frame
    known_solutions = [unique_solutions[x, :] for x in range(unique_solutions.shape[0])]
    # Calculate the mean Manhattan distance between the solutions - can use this as a scale reference to determine if any solutions in consecutive frames are new
    mean_manhattan = utils.mean_manhattan_distance_between_group_of_points(known_solutions)

    x_coords, y_coords = solve_interface.get_image_pixel_coords_wrapper(params.y_pixels, params.x_pixels, np.array(known_solutions))

    frame_idx = 0
    frame_idx_last_new_soln = 0
    total_duration = produce_image_timed(params.f_lambda, params.j_lambda, params.deltas[0],
                                          images_dir, params.colour_set, unique_solutions, x_coords, y_coords, 0)

    for delta in params.deltas[1:]:
        frame_idx += 1

        unique_solutions = solve_interface.find_unique_solutions_wrapper(params.f_lambda, params.j_lambda, delta, np.array(params.search_limits))
        # Add any new solutions to the known_solutions - new if it is not within 1% of the initial mean_manhattan of any existing solutions
        for j in range(unique_solutions.shape[0]):

            distances = [utils.manhattan_distance(known_solution, unique_solutions[j, :]) for known_solution in known_solutions]
            index_min = min(range(len(distances)), key=distances.__getitem__)

            if distances[index_min] < mean_manhattan/200:
                known_solutions[index_min] = unique_solutions[j, :]
            else:
                if frame_idx-frame_idx_last_new_soln < cfg.MIN_FRAMES_BETWEEN_NEW_SOLUTIONS:
                    # TODO: log then break out of loop here logger.debug(params.model_dump_json(exclude={'f_lambda', 'j_lambda'}))
                    raise Exception(f'A new solution was found in just {frame_idx-frame_idx_last_new_soln} frames. Try a lower delta-per-frame to stabilse the animation.')
                frame_idx_last_new_soln = frame_idx
                known_solutions.append(unique_solutions[j, :])
                print(f'There are now {len(known_solutions)} known solutions')

        print(f'Now solving the grid for frame {frame_idx+1} of {len(params.deltas)} (delta={delta}) ({unique_solutions.shape[0]} unique solutions)...')
        total_duration += produce_image_timed(params.f_lambda, params.j_lambda, delta, images_dir, params.colour_set, np.array(known_solutions), x_coords, y_coords, frame_idx)
        utils.print_time_remaining_estimate(frame_idx, len(params.deltas), total_duration)

    imaging.image.rgb_to_mp4(images_dir, params.fps)
    logger.debug(f'ENABLE_AOT={cfg.ENABLE_AOT}, ENABLE_QUADTREES={cfg.ENABLE_QUADTREES}')
    logger.debug(f'Generation time: {total_duration} s')


def create_still_inner(uuid: str, params: types.StillParameters):
    utils.logger_setup(logger, utils.get_images_dir(uuid), 'still')
    # logger.debug(params.model_dump_json(exclude={'f_lambda', 'j_lambda'}))

    images_dir = utils.get_images_dir(uuid)
    utils.mkdir_if_nonexistent(images_dir)
    unique_solutions = solve_interface.find_unique_solutions_wrapper(params.f_lambda, params.j_lambda, 0, np.array(params.search_limits))
    if len(unique_solutions) < 2:
        raise Exception('There are no solutions!' if len(unique_solutions) == 0 else 'There is only 1 solution!')

    x_coords, y_coords = solve_interface.get_image_pixel_coords_wrapper(params.y_pixels, params.x_pixels, unique_solutions)
    generation_time = produce_image_timed(params.f_lambda, params.j_lambda, 0.0, images_dir, params.colour_set, unique_solutions, x_coords, y_coords, 0)

    logger.debug(f'ENABLE_AOT={cfg.ENABLE_AOT}, ENABLE_QUADTREES={cfg.ENABLE_QUADTREES}')
    logger.debug(f'Generation time: {generation_time} s')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
