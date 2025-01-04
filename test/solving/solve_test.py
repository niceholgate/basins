import src.solving.solve as solve
import test.resources.shared as shared

# import runpy
# from pathlib import Path
# import os
# import sys

# test_dir = Path(os.path.realpath(__file__)).parent
# build_dir = test_dir.parent/'build'
# sys.path.append(str(build_dir))
# try:
#     import basins_solver
# except:
#     pass

import numpy as np

# @pytest.fixture(scope="session", autouse=True)
# def build():
#     # Setup - compile the basins_solver module from solve.py
#     print('Building ')
#     runpy.run_module('src.solver')
#     yield
#     # Teardown


# if cfg.SHOW_UNIQUE_SOLUTIONS_AND_EXIT:
#     utils.plot_unique_solutions(unique_solns_arr)
#     sys.exit(0)


def test_points_approx_equal():
    p1 = np.array([1.0, 1.0])
    p2 = np.array([4.0, 5.0])
    assert solve.points_approx_equal(p1, p2, 3.0)
    assert not solve.points_approx_equal(p1, p2, 2.0)


def test_newton_solve():
    soln, iters = solve.newton_solve(shared.LAMBDA_F, shared.LAMBDA_J, np.array([1.0, 2.0]), 0.0)
    assert solve.points_approx_equal(soln, np.array([0.64838359, 1.20401283]))
    assert iters == 4

