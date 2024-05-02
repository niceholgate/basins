import src.utils as utils
from src.solver import Solver


TEST_EXPRESSIONS = ['y**2+2*x*y+2*x**2-4-0.5*sin(15*(x+d))', 'y-10*x**2+3+d']
LAMBDA_F, LAMBDA_J = utils.get_lambdas(TEST_EXPRESSIONS)


def test_solve_grid():
    sut = Solver(LAMBDA_F, LAMBDA_J, 2, 3, 1)
    sut.solve_grid()
    a=2
