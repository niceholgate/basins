import src.utils as utils

import sys
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel, Field, field_validator
from typing import List, Callable


class StillRequest(BaseModel):
    colour_set: int = 1
    x_pixels: int = 500
    y_pixels: int = 500
    expressions: List[str]


class AnimationRequest(StillRequest):
    delta: int
    frames: int
    fps: int


class StillParameters(BaseModel):
    colour_set: int
    y_pixels: int = Field(gt=0)
    x_pixels: int = Field(gt=0)
    # Check successful parsing
    # Check only contains known symbols
    f_lambda: Callable
    j_lambda: Callable

    @staticmethod
    def from_request(request: StillRequest) -> 'StillParameters':
        # Sympy computes the partial derivatives of each equation with respect to x and y to obtain Jacobian matrix,
        # then "lambdifies" them into Python functions with position args x, y, d.
        f_lambda, j_lambda = utils.get_lambdas(request.expressions)
        return StillParameters(
            colour_set=request.colour_set,
            x_pixels=request.x_pixels,
            y_pixels=request.y_pixels,
            f_lambda=f_lambda,
            j_lambda=j_lambda)


class AnimationParameters(StillParameters):
    deltas: List[float]
    fps: int = Field(gt=0)

    @field_validator('deltas')
    @classmethod
    def require_multiple_frames(cls, v: List[float]) -> List[float]:
        if len(v) < 2:
            raise ValueError('Must request multiple frames')
        return v

    @staticmethod
    def from_request(request: AnimationRequest) -> 'AnimationParameters':
        # Terminate with error if system of equations does not include a d term
        f_sym = [parse_expr(ex) for ex in request.expressions]
        if all([SYMBOLS[2] not in exp.free_symbols for exp in f_sym]):
            print('For animations, must include at least one "d" term (delta to perturb the equation solutions)')
            sys.exit(0)
        return AnimationParameters(
            **StillParameters.from_request(request).model_dump(),
            deltas=np.linspace(0, request.delta, request.frames),
            fps=request.fps)


if __name__ == '__main__':
    AnimationRequest(delta=1, frames=2, fps=5)
