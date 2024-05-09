import src.utils as utils
import src.solver as solver

import numpy as np


TEST_EXPRESSIONS = ['y**2+2*x*y+2*x**2-4-0.5*sin(15*(x+d))', 'y-10*x**2+3+d']
LAMBDA_F, LAMBDA_J = utils.get_lambdas(TEST_EXPRESSIONS)

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


def test_solve_grid():
    sut = solver.Solver(LAMBDA_F, LAMBDA_J, 2, 3, 1)
    sut.solve_grid()
    assert sut.unique_solutions.shape == (4, 2)
    assert (sut.solutions_grid == np.array([[2, 3], [2, 4], [1, 4]])).all()
    assert (sut.iterations_grid == np.array([[7, 7], [7, 6], [7, 7]])).all()

