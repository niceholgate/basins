import config as cfg

from datetime import datetime
from pathlib import Path


def get_images_dir():
    images_dir = Path().cwd() / f'images/{cfg.start_time.strftime("%Y-%m-%d-%H-%M-%S")}'
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    return images_dir


def timed(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        return (datetime.now() - start).total_seconds()
    return wrapper


def print_time_remaining_estimate(i, N, duration_so_far):
    if i != N - 1:
        mean_duration = '{:.2f}'.format(duration_so_far / (i + 1))
        est_mins_remaining = '{:.2f}'.format(duration_so_far / (i + 1) * (N - i - 1) / 60)
        print(f'Mean frame generation time is {mean_duration} seconds;'
              f' estimate {est_mins_remaining} minutes remaining for video generation')