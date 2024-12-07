import logging

import src.config as cfg

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List


def mkdir_if_nonexistent(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)


def get_images_dir(uuid: str) -> Path:
    images_dir = Path(__file__).parent.parent.resolve() / f'images/{uuid}'
    return images_dir


def get_frame_filename(frame_number: int, file_extention: str) -> str:
    frame_number_formatted = cfg.FRAME_COUNT_PADDING.format(frame_number)
    return f'frame-{frame_number_formatted}.{file_extention}'


def timed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        return (datetime.now() - start).total_seconds()
    return wrapper





def print_time_remaining_estimate(i: int, N: int, duration_so_far: float) -> None:
    if i != N - 1:
        mean_time_per_frame = duration_so_far / (i + 1)
        mean_duration = '{:.2f}'.format(mean_time_per_frame)
        est_mins_remaining = '{:.2f}'.format(mean_time_per_frame * (N - i - 1) / 60)
        print(f'Mean frame generation time is {mean_duration} seconds;'
              f' estimate {est_mins_remaining} minutes remaining for video generation')


def plot_unique_solutions(unique_solns: npt.NDArray) -> None:
    plt.figure()
    plt.scatter(unique_solns[:, 0], unique_solns[:, 1])
    plt.show(block=True)


def logger_setup(logger: logging.Logger, images_dir: Path, file_name_base: str) -> None:
    logger.setLevel(logging.DEBUG)
    logs_files = list(images_dir.glob(f'{file_name_base}_*.log'))
    if logs_files:
        file_name = logs_files[0]
    else:
        mkdir_if_nonexistent(images_dir)
        datetimestr = datetime.utcnow().astimezone(timezone.utc).isoformat()\
            .replace(':', '-').replace('.', '-')
        file_name = f'{file_name_base}_{datetimestr}.log'
    # create file handler which logs even debug messages
    fh = logging.FileHandler(str(images_dir / file_name))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(u'[%(asctime)s] [%(threadName)s] [] [%(levelname)s] [%(lineno)d:%(filename)s(%(process)d)] - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def manhattan_distance(array1: npt.NDArray, array2: npt.NDArray):
    return np.abs(array1 - array2).sum()


def mean_manhattan_distance_between_group_of_points(points: List[npt.NDArray]):
    total = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            total += manhattan_distance(points[i], points[j])
    return total/(n * (n-1)/2)
