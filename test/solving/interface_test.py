import src.solving.interface as interface
import test.resources.shared as shared

from datetime import datetime
import pytest
import numpy as np


delta = 1.0
x_pixels = 67
y_pixels = 89
unique_solutions = interface.find_unique_solutions_wrapper(shared.LAMBDA_F, shared.LAMBDA_J, delta)
x_coords, y_coords = interface.get_image_pixel_coords(y_pixels, x_pixels, unique_solutions)


@pytest.fixture(scope='module')
def solve_times():
    solve_times = {}
    yield solve_times


def test_find_unique_solutions():
    assert unique_solutions.shape == (4, 2)


def test_get_image_pixel_coords():
    assert x_coords.shape == (67,)
    assert all(np.diff(x_coords) - 0.09341968 < 0.00001)
    assert y_coords.shape == (89,)
    assert all(np.diff(y_coords)-0.23465326 < 0.00001)


def test_solve_grid(solve_times):
    start = datetime.now()
    solutions, iterations = interface.solve_grid(shared.LAMBDA_F, shared.LAMBDA_J, x_coords, y_coords, delta, unique_solutions)
    solve_times['solve_grid'] = (datetime.now() - start).total_seconds()
    print(solve_times)

    assert unique_solutions.shape == (4, 2)
    assert np.abs(solutions.sum() - 15379) < 10
    assert np.abs(iterations.sum() - 38472) < 50


def test_solve_grid_quadtrees(solve_times):
    start = datetime.now()
    solutions, iterations = interface.solve_grid_quadtrees(shared.LAMBDA_F, shared.LAMBDA_J, x_coords, y_coords, delta, unique_solutions)
    solve_times['solve_grid'] = (datetime.now() - start).total_seconds()
    print(solve_times)

    assert unique_solutions.shape == (4, 2)
    assert np.abs(solutions.sum() - 15379) < 10
    assert np.abs(iterations.sum() - 36416) < 50
