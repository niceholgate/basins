import src.config as cfg
import src.calc as calc
import src.imaging as imaging
import src.types as types
import src.utils as utils

import sys
from datetime import datetime


@utils.timed
def produce_image_timed(images_dir, colour_set, unique_solns, x_coords, y_coords, f_lambda, j_lambda, delta, i):
    solutions, iterations = calc.solve_grid(unique_solns, x_coords, y_coords, f_lambda, j_lambda, delta)
    imaging.save_still(images_dir, solutions, iterations, unique_solns, y_coords.shape[0], x_coords.shape[0],
                       smoothing=False, blending=False, colour_set=colour_set, frame=i)

# TODO:
#  1. add a CLI
#  2. add logging
#  3. Consolidate input validations in one place
#  4. Animations can pan/zoom the grid
#  5. queue and RL the requests
#  6. save inputs with images


def create_animation(uuid: str, params: types.AnimationParameters):
    images_dir = utils.get_images_dir(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), uuid)

    first_frame_unique_solns = calc.find_unique_solutions(params.f_lambda, params.j_lambda, params.deltas[0])
    x_coords, y_coords = calc.get_image_pixel_coords(params.y_pixels, params.x_pixels, first_frame_unique_solns)

    # Terminate with error if system of equations does not include a d term
    if all([cfg.symbols[2] not in exp.free_symbols for exp in cfg.f_sym]):
        print('For animations, must include at least one "d" term (delta to perturb the equation solutions)')
        sys.exit(0)
    # Assume that if the same number of solutions is found each time, the sorted solutions will
    # correspond to each other in sequence between different deltas
    unique_solns_per_delta = [first_frame_unique_solns]
    expected_number_of_solns = len(unique_solns_per_delta[0])
    for delta in params.deltas[1:]:
        this_delta_unique_solns = calc.find_unique_solutions(params.f_lambda, params.j_lambda, delta)
        if len(this_delta_unique_solns) > expected_number_of_solns:
            print(f'Terminating because number of solutions increased from {expected_number_of_solns}'
                  f' to {len(this_delta_unique_solns)} for delta={delta}')
            sys.exit(0)
        unique_solns_per_delta.append(this_delta_unique_solns)
    total_duration = 0.0
    for i, delta in enumerate(params.deltas):
        print(f'Now solving the grid for frame {i + 1} of {len(params.deltas)} (delta={delta})...')
        total_duration += produce_image_timed(images_dir, params.colour_set, unique_solns_per_delta[i], x_coords, y_coords, params.f_lambda, params.j_lambda, delta, i)
        utils.print_time_remaining_estimate(i, len(params.deltas), total_duration)
    imaging.stills_to_video(images_dir, params.fps)


def create_still(uuid: str, params: types.StillParameters):
    images_dir = utils.get_images_dir(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), uuid)

    first_frame_unique_solns = calc.find_unique_solutions(params.f_lambda, params.j_lambda, 0)
    x_coords, y_coords = calc.get_image_pixel_coords(params.y_pixels, params.x_pixels, first_frame_unique_solns)
    produce_image_timed(images_dir, params.colour_set, first_frame_unique_solns, x_coords, y_coords, params.f_lambda, params.j_lambda, 0, 0)

