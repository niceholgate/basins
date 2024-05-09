from src.solver import Solver
import src.imaging as imaging
import src.types as types
import src.utils as utils

from sys import exit
from datetime import datetime


@utils.timed
def produce_image_timed(solver: Solver, images_dir, colour_set, i):
    solver.solve_grid()
    imaging.save_still(images_dir, solver, smoothing=False, blending=False, colour_set=colour_set, frame=i)

# TODO:
# -create a front end
# -add a CLI
# -add logging
# -Consolidate input validations in one place
# -Animations can pan/zoom the grid
# -queue and RL the requests
# -save inputs with images


def create_animation(uuid: str, params: types.AnimationParameters):
    images_dir = utils.get_images_dir(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), uuid)
    solvers = [Solver(params.f_lambda, params.j_lambda, params.y_pixels, params.x_pixels, params.deltas[0])]
    expected_number_of_solns = solvers[0].unique_solutions.shape[0]

    # Assume that if the same number of solutions is found each time, the sorted solutions will
    # correspond to each other in sequence between different deltas
    for delta in params.deltas[1:]:
        solvers.append(Solver(params.f_lambda, params.j_lambda, params.y_pixels, params.x_pixels, delta))
        if solvers[-1].unique_solutions.shape[0] > expected_number_of_solns:
            print(f'Terminating because number of solutions increased from {expected_number_of_solns}'
                  f' to {solvers[-1].unique_solutions.shape[0]} for delta={delta}')
            exit(0)

    total_duration = 0.0
    for i, solver in enumerate(solvers):
        print(f'Now solving the grid for frame {i + 1} of {len(solvers)} (delta={solver.delta})...')
        total_duration += produce_image_timed(solver, images_dir, params.colour_set, i)
        utils.print_time_remaining_estimate(i, len(params.deltas), total_duration)
    imaging.stills_to_video(images_dir, params.fps)


def create_still(uuid: str, params: types.StillParameters):
    images_dir = utils.get_images_dir(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), uuid)
    solver = Solver(params.f_lambda, params.j_lambda, params.y_pixels, params.x_pixels, 0)

    produce_image_timed(solver, images_dir, params.colour_set, 0)

