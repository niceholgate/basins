import src.utils as sut
import numpy as np
from time import sleep


def test_manhattan_distance():
    assert sut.manhattan_distance(np.array([-4, 1]), np.array([6, 3])) == 10 + 2


def test_mean_manhattan_distance_between_group_of_points():
    points = [np.array([0, 0]), np.array([2, 3]), np.array([-1, -1]), np.array([1, 1])]
    assert sut.mean_manhattan_distance_between_group_of_points(points) == (5 + 2 + 7 + 2 + 3 + 4)/6


def test_timed():
    timed_sleep = sut.timed(sleep)
    time_taken = timed_sleep(0.1)
    assert abs(time_taken-0.1) < 0.01
