import src.api.requests as requests


TEST_EXPRESSIONS = ['y**2+2*x*y+2*x**2-4-0.5*sin(15*(x+d))', 'y-10*x**2+3+d']
LAMBDA_F, LAMBDA_J = requests.StillParameters.get_lambdas(TEST_EXPRESSIONS)