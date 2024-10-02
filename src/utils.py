import logging

import src.config as cfg

import numpy.typing as npt
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


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


def logger_setup(logger: logging.Logger, uuid: str, file_name_base: str) -> None:
    logger.setLevel(logging.DEBUG)
    directory = get_images_dir(uuid)
    logs_files = list(directory.glob(f'{file_name_base}_*.log'))
    if logs_files:
        file_name = logs_files[0]
    else:
        mkdir_if_nonexistent(directory)
        datetimestr = datetime.utcnow().astimezone(timezone.utc).isoformat()\
            .replace(':', '-').replace('.', '-')
        file_name = f'{file_name_base}_{datetimestr}.log'
    # create file handler which logs even debug messages
    fh = logging.FileHandler(str(directory / file_name))
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
