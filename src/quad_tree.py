from typing import Tuple
import numpy as np

class QuadTree:
    x_lims: Tuple[int, int]
    y_lims: Tuple[int, int]
    nw: 'QuadTree'
    ne: 'QuadTree'
    sw: 'QuadTree'
    se: 'QuadTree'

    def __init__(self, x_lims: Tuple[int, int], y_lims: Tuple[int, int]):
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.boundary_coordinates_generator = self.get_boundary_coordinates_generator()

    def get_boundary_coordinates_generator(self):
        for x in range(self.x_lims[0], self.x_lims[1] + 1):
            yield x, self.y_lims[0]
        for y in range(self.y_lims[0] + 1, self.y_lims[1] + 1):
            yield self.x_lims[1], y
        for x in reversed(range(self.x_lims[0], self.x_lims[1])):
            yield x, self.y_lims[1]
        for y in reversed(range(self.y_lims[0]+1, self.y_lims[1])):
            yield self.x_lims[0], y

    def iterate_boundary_coordinates(self):
        return next(self.boundary_coordinates_generator)

    # def subdivide(self):
    #     x_mid = np.ceil(np.mean(self.x_lims))
    #     y_mid = np.ceil(np.mean(self.y_lims))
    #
    #     # Cannot subdivide a point
    #     if self.x_lims[0] == self.x_lims[1] and self.y_lims[0] == self.y_lims[1]:
    #         return
    #     # For a single column, can only subdivide in y direction
    #     if self.x_lims[0] == self.x_lims[1]:
    #         self.nw = QuadTree((x_mid, x_mid), (self.y_lims[0], y_mid))
    #         self.sw = QuadTree()
    #     # For a single row, can only subdivide in x direction
    #
    #
    #     self.nw = QuadTree((self.x_lims[0], x_mid), (self.y_lims[0], y_mid))
    #     self.ne = QuadTree