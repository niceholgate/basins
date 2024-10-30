import src.config as cfg

import numpy as np
# import numpy.typing as npt
import sympy as sp
import numba as nb
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel, Field, field_validator
from typing import List, Callable, Tuple
import taichi.math as tm
import taichi as ti

SYMBOLS = sp.symbols('x y d')


class StillRequest(BaseModel):
    colour_set: int = 1
    x_pixels: int = 500
    y_pixels: int = 500
    expressions: List[str]
    search_limits: List[float]


class AnimationRequest(StillRequest):
    delta: float
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
    search_limits: List[float]

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
            j_lambda=j_lambda,
            search_limits=request.search_limits)

    @staticmethod
    def get_lambdas(expressions: List[str]) -> Tuple[Callable, Callable]:
        """
        # Sympy computes the partial derivatives of each equation with respect to x and y to obtain Jacobian matrix,
        # then "lambdifies" them into Python functions with position args x, y, d.
        """

        # - The strings ``"math"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``,
        #   ``"scipy"``, ``"sympy"``, or ``"tensorflow"`` or ``"jax"``. This uses the
        #   corresponding printer and namespace mapping for that module.
        # - A module (e.g., ``math``). This uses the global namespace of the
        #   module. If the module is one of the above known modules, it will
        #   also use the corresponding printer and namespace mapping
        #   (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).
        # - A dictionary that maps names of SymPy functions to arbitrary
        #   functions
        #   (e.g., ``{'sin': custom_sin}``).
        # - A list that contains a mix of the arguments above, with higher
        #   priority given to entries appearing first
        #   (e.g., to use the NumPy module but override the ``sin`` function
        #   with a custom version, you can use
        #   ``[{'sin': custom_sin}, 'numpy']``).

        f_sym = [sp.parsing.sympy_parser.parse_expr(ex) for ex in expressions]
        j_sym = [[sp.diff(exp, sym) for sym in SYMBOLS[:2]] for exp in f_sym]
        # f_lambda = nb.njit(sp.lambdify(SYMBOLS, f_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
        # j_lambda = nb.njit(sp.lambdify(SYMBOLS, j_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
        f_lambda = sp.lambdify(SYMBOLS, f_sym, [{'sin': tm.sin, 'cos': tm.cos}, 'numpy'])
        j_lambda = sp.lambdify(SYMBOLS, j_sym, [{'sin': tm.sin, 'cos': tm.cos}, 'numpy'])
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
