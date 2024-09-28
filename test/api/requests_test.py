import src.api.requests as requests
import test.resources.shared as shared

import numpy as np
import pytest


def test_get_lambdas():
    f, j = requests.StillParameters.get_lambdas(shared.TEST_EXPRESSIONS)
    f_num = np.array(f(1.0, 2.0, 3.0), dtype=np.float_)
    j_num = np.array(j(1.0, 2.0, 3.0), dtype=np.float_)
    assert f_num.shape == (2,)
    assert j_num.shape == (2, 2)


def test_animation_parameters_from_request_no_delta_term_raises_value_error():
    request = requests.AnimationRequest(expressions=['y**2+2*x*y+2*x**2-4-0.5*sin(15*(x))', 'y-10*x**2+3'],
                                        delta=1.0,
                                        frames=10,
                                        fps=5)
    with pytest.raises(ValueError) as excinfo:
        params = requests.AnimationParameters.from_request(request)
    assert excinfo.value.args[0] == 'For animations, must include at least one "d" term (delta to perturb the equation solutions)'