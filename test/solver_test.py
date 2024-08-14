import src.utils as utils
import src.solver as solver

import basins_solver

import numpy as np
import pytest

from datetime import datetime


@pytest.fixture(scope='module')
def solve_times():
    solve_times = {}
    yield solve_times


TEST_EXPRESSIONS = ['y**2+2*x*y+2*x**2-4-0.5*sin(15*(x+d))', 'y-10*x**2+3+d']
LAMBDA_F, LAMBDA_J = utils.get_lambdas(TEST_EXPRESSIONS)
SOLVE_GRID_TIME = None

# if cfg.SHOW_UNIQUE_SOLUTIONS_AND_EXIT:
#     utils.plot_unique_solutions(unique_solns_arr)
#     sys.exit(0)


def test_get_lambdas():
    f, j = utils.get_lambdas(TEST_EXPRESSIONS)
    f_num = np.array(f(1.0, 2.0, 3.0), dtype=np.float_)
    j_num = np.array(j(1.0, 2.0, 3.0), dtype=np.float_)
    assert f_num.shape == (2,)
    assert j_num.shape == (2, 2)


def test_newton_solve():
    soln, iters = solver.newton_solve(LAMBDA_F, LAMBDA_J, np.array([1.0, 2.0]), 0.0)
    assert solver.points_approx_equal(soln, np.array([0.64838359, 1.20401283]))
    assert iters == 4


# def test_solve_grid(solve_times):
#     sut = solver.Solver(LAMBDA_F, LAMBDA_J, 67, 89, 1.0)
#
#     assert sut.unique_solutions.shape == (4, 2)
#
#     start = datetime.now()
#     sut.solve_grid()
#     solve_times['solve_grid'] = (datetime.now() - start).total_seconds()
#     print(solve_times)
#
#     assert np.abs(sut.solutions_grid.sum() - 15379) < 10
#     assert np.abs(sut.iterations_grid.sum() - 38472) < 50


# def test_solve_grid_quadtrees(solve_times):
#     sut = solver.Solver(LAMBDA_F, LAMBDA_J, 67, 89, 1.0)
#
#     assert sut.unique_solutions.shape == (4, 2)
#
#     start = datetime.now()
#     sut.solve_grid_quadtrees()
#     solve_times['solve_grid_quadtrees'] = (datetime.now() - start).total_seconds()
#     print(solve_times)
#
#     assert np.abs(sut.solutions_grid.sum() - 15379) < 10
#     assert np.abs(sut.iterations_grid.sum() - 36416) < 50


def test_solve_grid():
    delta = 1.0
    x_pixels = 67
    y_pixels = 89

    unique_solutions = basins_solver.find_unique_solutions(LAMBDA_F, LAMBDA_J, delta)

    x_coords, y_coords = basins_solver.get_image_pixel_coords(y_pixels, x_pixels, unique_solutions)

    solutions, iterations = basins_solver.solve_grid(LAMBDA_F, LAMBDA_J, x_coords, y_coords, delta, unique_solutions)

    assert unique_solutions.shape == (4, 2)

    # start = datetime.now()
    # sut.solve_grid_quadtrees()
    # # solve_times['solve_grid_quadtrees'] = (datetime.now() - start).total_seconds()
    # print(solve_times)

    assert np.abs(solutions.sum() - 15379) < 10
    assert np.abs(iterations.sum() - 36416) < 50

