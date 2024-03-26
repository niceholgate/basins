import src.config as cfg

import numpy as np
import sympy as sp
import numba as nb
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel, Field, field_validator#, conlist
from typing import List, Callable


class StillRequest(BaseModel):
    colour_set: int = 1
    x_pixels: int = 500
    y_pixels: int = 500
    expressions: List[str]
    # expressions: List[str] = conlist(str, min_length=2, max_length=2)


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
        symbols = sp.symbols('x y d')
        f_sym = [parse_expr(ex) for ex in request.expressions]
        j_sym = [[sp.diff(exp, sym) for sym in symbols[:2]] for exp in f_sym]
        f_lambda = nb.njit(sp.lambdify(symbols, f_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
        j_lambda = nb.njit(sp.lambdify(symbols, j_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)

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
        return AnimationParameters(
            **StillParameters.from_request(request).model_dump(),
            deltas=np.linspace(0, request.delta, request.frames),
            fps=request.fps)


if __name__ == '__main__':
    AnimationRequest(delta=1, frames=2, fps=5)
