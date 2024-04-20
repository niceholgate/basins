from typing import Tuple, Optional
import numpy as np


class QuadTree:
    x_lims: Tuple[int, int]
    y_lims: Tuple[int, int]
    nw: Optional['QuadTree'] = None
    ne: Optional['QuadTree'] = None
    sw: Optional['QuadTree'] = None
    se: Optional['QuadTree'] = None

    def __init__(self, x_lims: Tuple[int, int], y_lims: Tuple[int, int]):
        self.x_lims = x_lims
        self.y_lims = y_lims

    def boundary_coordinates_generator(self):
        for x in range(self.x_lims[0], self.x_lims[1] + 1):
            yield x, self.y_lims[0]
        for y in range(self.y_lims[0] + 1, self.y_lims[1] + 1):
            yield self.x_lims[1], y
        for x in reversed(range(self.x_lims[0], self.x_lims[1])):
            yield x, self.y_lims[1]
        for y in reversed(range(self.y_lims[0]+1, self.y_lims[1])):
            yield self.x_lims[0], y

    def interior_coordinates_generator(self):
        for x in range(self.x_lims[0] + 1, self.x_lims[1]):
            for y in range(self.y_lims[0] + 1, self.y_lims[1]):
                yield x, y

    def random_interior_coordinates(self):
        return np.random.randint(self.x_lims[0] + 1, self.x_lims[1]), np.random.randint(self.y_lims[0] + 1, self.y_lims[1])

    def subdivide(self):
        x_mid = int(np.floor(np.mean(self.x_lims)))
        y_mid = int(np.floor(np.mean(self.y_lims)))

        # Cannot subdivide a point
        if self.x_lims[0] == self.x_lims[1] and self.y_lims[0] == self.y_lims[1]:
            return
        # For a single column, can only subdivide in y direction
        if self.x_lims[0] == self.x_lims[1]:
            self.nw = QuadTree((x_mid, x_mid), (self.y_lims[0], y_mid))
            self.sw = QuadTree((x_mid, x_mid), (y_mid+1, self.y_lims[1]))
            return
        # For a single row, can only subdivide in y direction
        if self.y_lims[0] == self.y_lims[1]:
            self.nw = QuadTree((self.x_lims[0], x_mid), (y_mid, y_mid))
            self.ne = QuadTree((x_mid+1, self.x_lims[1]), (y_mid, y_mid))
            return
        self.nw = QuadTree((self.x_lims[0], x_mid), (self.y_lims[0], y_mid))
        self.ne = QuadTree((x_mid+1, self.x_lims[1]), (self.y_lims[0], y_mid))
        self.sw = QuadTree((self.x_lims[0], x_mid), (y_mid+1, self.y_lims[1]))
        self.se = QuadTree((x_mid+1, self.x_lims[1]), (y_mid+1, self.y_lims[1]))

    def get_children(self):
        return [child for child in [self.nw, self.ne, self.sw, self.se] if child is not None]

    def __eq__(self, other: 'QuadTree'):
        return self.x_lims == other.x_lims and self.y_lims == other.y_lims
