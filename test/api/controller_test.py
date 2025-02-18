import src.api.controller as sut
import src.api.requests as types
import test.resources.shared as shared

import pytest
from httpx import AsyncClient

from fastapi import BackgroundTasks, Response
# from pydantic import ValidationError


@pytest.mark.anyio
async def test_create_still_happy_path(mocker):
    request = types.StillRequest(expressions=shared.TEST_EXPRESSIONS, search_limits=[-1000, 1000, -1000, 1000])
    mocker.patch('src.api.controller.create_still_inner', None)

    async with AsyncClient(app=sut.app, base_url="http://test") as ac:
        response = await sut.create_still(request, Response(), BackgroundTasks())
    assert response.body['message'] == 'Creating an image'
    assert 'id' in response.body

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
                                     search_limits=[-1000, 1000, -1000, 1000],
                                     delta=1.0,
                                     frames=10,
                                     fps=5)
    mocker.patch('src.api.controller.create_animation_inner', None)

    async with AsyncClient(app=sut.app, base_url="http://test") as ac:
        response = await sut.create_animation(request, Response(), BackgroundTasks())
    assert response.body['message'] == 'Creating an animation'
    assert 'id' in response.body


# TODO: add more validation to the animation request parameters
@pytest.mark.anyio
async def test_create_animation_error_on_single_frame(mocker):
    request = types.AnimationRequest(expressions=shared.TEST_EXPRESSIONS,
                                     search_limits=[-1000, 1000, -1000, 1000],
                                     delta=1.0,
                                     frames=1,
                                     fps=5)
    mocker.patch('src.api.controller.create_animation_inner', None)

    async with AsyncClient(app=sut.app, base_url="http://test") as ac:
        response = await sut.create_animation(request, Response(), BackgroundTasks())
    assert response.status_code == 400
    assert response.body['message'] == "Input errors: {'deltas': 'Value error, Must request multiple frames'}"
    assert 'id' not in response.body


