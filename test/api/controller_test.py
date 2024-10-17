import src.api.controller as sut
import src.api.requests as types
import test.resources.shared as shared
import numpy as np

import pytest
from httpx import AsyncClient

from fastapi import BackgroundTasks, Response
# from pydantic import ValidationError


@pytest.mark.anyio
async def test_create_still_happy_path(mocker):
    request = types.StillRequest(expressions=shared.TEST_EXPRESSIONS)
    mocker.patch('src.api.controller.create_still_inner', None)

    async with AsyncClient(app=sut.app, base_url="http://test") as ac:
        response = await sut.create_still(request, Response(), BackgroundTasks())
    assert response['message'] == 'Creating an image'
    assert 'id' in response

# TODO: add validation to the still request parameters
# @pytest.mark.anyio
# async def test_create_still_validation_error(mocker):
#     request = types.StillRequest(colour_set=-1, expressions=shared.TEST_EXPRESSIONS)
# # # mocker.patch("os.remove", side_effect=FileNotFoundError)
#
#     with pytest.raises(ValidationError):
#         async with AsyncClient(app=sut.app, base_url="http://test") as ac:
#             response = await sut.create_still(request, Response(), BackgroundTasks())
#     assert 'Input errors:' in response['message']

@pytest.mark.anyio
async def test_create_animation_happy_path(mocker):
    request = types.AnimationRequest(expressions=shared.TEST_EXPRESSIONS,
                                     delta=1.0,
                                     frames=10,
                                     fps=5)
    mocker.patch('src.api.controller.create_animation_inner', None)

    async with AsyncClient(app=sut.app, base_url="http://test") as ac:
        response = await sut.create_animation(request, Response(), BackgroundTasks())
    assert response['message'] == 'Creating an animation'
    assert 'id' in response


# TODO: add more validation to the animation request parameters
# TODO: temporal interpolation (create frames spread evenly through the whole animation then interpolate certain pixels for the remaining frames
@pytest.mark.anyio
async def test_create_animation_happy_path(mocker):
    request = types.AnimationRequest(expressions=shared.TEST_EXPRESSIONS,
                                     delta=1.0,
                                     frames=1,
                                     fps=5)
    mocker.patch('src.api.controller.create_animation_inner', None)

    async with AsyncClient(app=sut.app, base_url="http://test") as ac:
        response = await sut.create_animation(request, Response(), BackgroundTasks())
    assert response['message'] == "Input errors: {'deltas': 'Value error, Must request multiple frames'}"
    assert 'id' not in response


def test_manhattan_distance():
    assert sut.manhattan_distance(np.array([-4, 1]), np.array([6, 3])) == 10 + 2


def test_mean_manhattan_distance_between_group_of_points():
    points = [np.array([0, 0]), np.array([2, 3]), np.array([-1, -1]), np.array([1, 1])]
    assert sut.mean_manhattan_distance_between_group_of_points(points) == (5 + 2 + 7 + 2 + 3 + 4)/6