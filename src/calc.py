import src.config as cfg
import src.utils as utils
from src.quad_tree import QuadTree

import numpy as np
import numpy.typing as npt
import numba as nb
import sys
from typing import Tuple, Callable, List

nb.config.DISABLE_JIT = cfg.DISABLE_JIT


class Solver:

    def __init__(self, f_lambda: Callable, j_lambda: Callable, y_coords: npt.NDArray, x_coords: npt.NDArray, delta: float):
        self._f_lambda = f_lambda
        self._j_lambda = j_lambda
        self.y_coords = y_coords
        self.x_coords = x_coords
        self.delta = delta
        self._unique_solutions = self._find_unique_solutions()
        self.solutions_local = np.zeros((y_coords.shape[0], x_coords.shape[0]), dtype=np.int_)
        self.iterations_local = np.zeros((y_coords.shape[0], x_coords.shape[0]), dtype=np.int_)

    # TODO: delta only needed as an input for each frame solution?
    def set_delta(self, delta):
        self.delta = delta
        self._unique_solutions = self._find_unique_solutions()

    # @nb.njit(target_backend=cfg.NUMBA_TARGET)
    def _find_unique_solutions(self) -> npt.NDArray:
        """Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found."""
        unique_solns: List[npt.NDArray] = []
        x_randoms = cfg.SEARCH_X_LIMS[0] + (cfg.SEARCH_X_LIMS[1]-cfg.SEARCH_X_LIMS[0])*np.random.random(cfg.MAX_SEARCH_POINTS)
        y_randoms = cfg.SEARCH_Y_LIMS[0] + (cfg.SEARCH_Y_LIMS[1]-cfg.SEARCH_Y_LIMS[0])*np.random.random(cfg.MAX_SEARCH_POINTS)
        randoms = np.array([x_randoms, y_randoms])
        point_count = 0
        converged_search_points_since_last_new_soln = 0
        for idx in range(randoms.shape[1]):
            point_count += 1
            soln, iters = self.newton_solve(self._f_lambda, self._j_lambda, randoms[:, idx], self.delta)
            if iters < cfg.MAX_ITERS:
                if not unique_solns:
                    unique_solns.append(soln)
                if any([self._points_approx_equal(existing_soln, soln) for existing_soln in unique_solns]):
                    converged_search_points_since_last_new_soln += 1
                else:
                    converged_search_points_since_last_new_soln = 0
                    unique_solns.append(soln)
                if converged_search_points_since_last_new_soln >= cfg.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION:
                    print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of '
                          f'{cfg.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points'
                          f' since the last new unique solution.')
                    break

        # Temporarily convert the solutions to tuples to sort them (ensures the random search returns the same result each
        # time for a given system of equations) then put them into one 2D array
        unique_solns_arr = np.array(sorted([tuple(s) for s in unique_solns]))

        if len(unique_solns) < 2:
            print('Found fewer than 2 unique solutions, cannot generate an image')
            sys.exit(0)

        if cfg.SHOW_UNIQUE_SOLUTIONS_AND_EXIT:
            utils.plot_unique_solutions(unique_solns_arr)
            sys.exit(0)

        return unique_solns_arr


    # @nb.njit(target_backend=cfg.NUMBA_TARGET)
    @staticmethod
    def get_image_pixel_coords(y_pixels: int, x_pixels: int, unique_solns: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get the coordinates of all the pixels in the image, ensuring that all the known solutions to the system of
        equations fall safely within the bounds of the image."""
        # Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
        x_min, x_mean, x_max = unique_solns[:, 0].min(), unique_solns[:, 0].mean(), unique_solns[:, 0].max()
        y_min, y_mean, y_max = unique_solns[:, 1].min(), unique_solns[:, 1].mean(), unique_solns[:, 1].max()

        # Need to handle cases where solutions are collinear in x or y directions
        if y_min == y_max:
            x_range = (x_max - x_min) * 4
            y_range = x_range * y_pixels / x_pixels
        elif x_min == x_max:
            y_range = (y_max - y_min) * 4
            x_range = y_range * x_pixels / y_pixels
        else:
            x_range = (x_max - x_min) * 4
            y_range = (y_max - y_min) * 4

        return np.linspace(x_mean-x_range/2, x_mean+x_range/2, x_pixels),\
               np.linspace(y_mean-y_range/2, y_mean+y_range/2, y_pixels)

    def get_index_of_matching_unique_soln(self, soln: npt.NDArray) -> int:
        match = -1
        for unique_soln_idx in range(self._unique_solutions.shape[0]):
            if self._points_approx_equal(self._unique_solutions[unique_soln_idx, :], soln):
                match = unique_soln_idx + 1
                break
        return match

    def set_pixel_values(self, j: int, i: int):
        if self.solutions_local[j, i] == 0:
            solution_pixel, iterations_pixel = self.newton_solve(self._f_lambda, self._j_lambda, np.array([self.x_coords[i], self.y_coords[j]]), self.delta)
            self.iterations_local[j, i] = iterations_pixel
            if iterations_pixel < cfg.MAX_ITERS:
                match = self.get_index_of_matching_unique_soln(solution_pixel)
                if match != -1:
                    self.solutions_local[j, i] = match
                else:
                    print(f'WARNING: Image will ignore a novel solution found on the grid: {solution_pixel}')


    @nb.njit(target_backend=cfg.NUMBA_TARGET)
    def solve_grid(self):
        """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""
        for j in range(self.y_coords.shape[0]):
            for i, x_init in enumerate(self.x_coords):
                self.set_pixel_values(j, i)

    @nb.njit(target_backend=cfg.NUMBA_TARGET)
    def solve_grid_quadtrees(self):
        """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""

        top_qt = QuadTree((0, self.x_coords.shape[0]), (0, self.y_coords.shape[0]), None)

        # Iterate around the boundary of the current QuadTree calculating the solution, and note if they are all identical
        qt = top_qt
        last_boundary_soln = None
        unique_boundary_soln = True
        for i, j in qt.boundary_coordinates_generator():
            soln, iters = self.newton_solve_caching(self._f_lambda, self._j_lambda, np.array([self.x_coords[i], self.y_coords[j]]), self.delta,
                                               self.solutions_local, self.iterations_local, i, j)
            self.set_pixel_values(j, i)
            if unique_boundary_soln:
                if last_boundary_soln:
                    unique_boundary_soln = self.solutions_local[j, i] == last_boundary_soln
                last_boundary_soln = self.solutions_local[j, i]

        # If they are all identical around the boundary, check some random interior points.
        # If those are all the same too, then fill the whole QuadTree area with that solution.
        if unique_boundary_soln:
            n_interior_pixels = max((qt.x_lims[1]-qt.x_lims[0]-1), 0)*max((qt.y_lims[1]-qt.y_lims[0]-1), 0)
            pixels_checked = 0
            while pixels_checked < 20 and pixels_checked < n_interior_pixels/2:
                rand_i, rand_j = qt.random_interior_coordinates()
                soln, iters = self.newton_solve_caching(self._f_lambda, self._j_lambda, np.array([self.x_coords[rand_i], self.y_coords[rand_j]]), self.delta,
                                                   self.solutions_local, self.iterations_local, rand_i, rand_j)
                self.set_pixel_values(rand_j, rand_i)
                unique_boundary_soln = self.solutions_local[rand_j, rand_i] == last_boundary_soln
                if not unique_boundary_soln:
                    break

            if unique_boundary_soln:
                self.solutions_local[qt.y_lims[0]:qt.y_lims[1]+1, qt.x_lims[0]:qt.x_lims[1]+1] = last_boundary_soln
                qt.terminal = True

    @nb.njit(target_backend=cfg.NUMBA_TARGET)
    def newton_solve_caching(self, f_lam: Callable, j_lam: Callable, starting_guess: npt.NDArray, d: float,
                             solutions: npt.NDArray, iterations: npt.NDArray, i: int, j: int) -> Tuple[npt.NDArray, int]:
        if solutions[j, i] == 0:
            soln, iters = self.newton_solve(f_lam, j_lam, starting_guess, d)
            solutions[j, i] = soln
            iterations[j, i] = iters
        return solutions[j, i], iterations[j, i]

    @staticmethod
    @nb.njit(target_backend=cfg.NUMBA_TARGET)
    def newton_solve(f_lam: Callable, j_lam: Callable, starting_guess: npt.NDArray, d: float) -> Tuple[npt.NDArray, int]:
        """Perform Newton's method until either the solution converges or the maximum iterations are exceeded."""
        current_guess = starting_guess
        # delta_norm_hist = []
        delta_norm = np.float_(1000.0)
        n_iters = 0
        while delta_norm > cfg.EPSILON and n_iters < cfg.MAX_ITERS:
            f_num = f_lam(current_guess[0], current_guess[1], d)
            j_num = j_lam(current_guess[0], current_guess[1], d)
            delta = np.linalg.solve(np.array(j_num), -np.array(f_num)).flatten()
            current_guess = current_guess + delta
            delta_norm = np.linalg.norm(delta)
            # delta_norm_hist.append(delta_norm)
            n_iters += 1

        return current_guess, n_iters

    @staticmethod
    @nb.njit(target_backend=cfg.NUMBA_TARGET)
    def _points_approx_equal(p1: npt.NDArray, p2: npt.NDArray) -> bool:
        return bool(np.linalg.norm(p1 - p2) < 2 * cfg.EPSILON)

