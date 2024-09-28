import src.utils as utils
import src.config as cfg

import sys
import numpy as np
import sympy as sp
import numba as nb
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel, Field, field_validator
from typing import List, Callable, Tuple


SYMBOLS = sp.symbols('x y d')


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
    expressions: List[str] # remove this?
    # Check successful parsing
    # Check only contains known symbols
    f_lambda: Callable
    j_lambda: Callable

    @staticmethod
    def from_request(request: StillRequest) -> 'StillParameters':
        # Sympy computes the partial derivatives of each equation with respect to x and y to obtain Jacobian matrix,
        # then "lambdifies" them into Python functions with position args x, y, d.
        f_lambda, j_lambda = StillParameters.get_lambdas(request.expressions)
        return StillParameters(
            colour_set=request.colour_set,
            x_pixels=request.x_pixels,
            y_pixels=request.y_pixels,
            expressions=request.expressions,
            f_lambda=f_lambda,
            j_lambda=j_lambda)

    @staticmethod
    def get_lambdas(expressions: List[str]) -> Tuple[Callable, Callable]:
        """
        # Sympy computes the partial derivatives of each equation with respect to x and y to obtain Jacobian matrix,
        # then "lambdifies" them into Python functions with position args x, y, d.
        """
        f_sym = [sp.parsing.sympy_parser.parse_expr(ex) for ex in expressions]
        j_sym = [[sp.diff(exp, sym) for sym in SYMBOLS[:2]] for exp in f_sym]
        f_lambda = nb.njit(sp.lambdify(SYMBOLS, f_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
        j_lambda = nb.njit(sp.lambdify(SYMBOLS, j_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
        return f_lambda, j_lambda


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
            raise ValueError('For animations, must include at least one "d" term (delta to perturb the equation solutions)')
        return AnimationParameters(
            **StillParameters.from_request(request).model_dump(),
            deltas=np.linspace(0, request.delta, request.frames).tolist(),
            fps=request.fps)


if __name__ == '__main__':
    AnimationRequest(delta=1, frames=2, fps=5)
