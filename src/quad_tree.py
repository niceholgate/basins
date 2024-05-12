from typing import Tuple, Optional, Generator
import numpy as np


class QuadTree:
    x_lims: Tuple[int, int]
    y_lims: Tuple[int, int]
    parent: Optional['QuadTree'] = None
    terminal: bool = False
    _nw: Optional['QuadTree'] = None
    _ne: Optional['QuadTree'] = None
    _sw: Optional['QuadTree'] = None
    _se: Optional['QuadTree'] = None
    _next_node_dfs_generator: Generator = None

    def __init__(self, x_lims: Tuple[int, int], y_lims: Tuple[int, int], parent: Optional['QuadTree']):
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.parent = parent
        if self.x_lims[0] == self.x_lims[1] and self.y_lims[0] == self.y_lims[1]:
            self.terminal = True

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

    def random_interior_coordinates(self) -> (int, int):
        return int(np.random.randint(self.x_lims[0] + 1, self.x_lims[1])),\
               int(np.random.randint(self.y_lims[0] + 1, self.y_lims[1]))

    def get_next_node_dfs(self):
        if self._next_node_dfs_generator is None:
            self._next_node_dfs_generator = self._create_next_node_dfs_generator()
        return next(self._next_node_dfs_generator)

    def _create_next_node_dfs_generator(self):
        for child in self.get_children():
            yield child
        while True:
            if self.parent is None:
                yield None
            yield self.parent.get_next_node_dfs()

    def get_children(self):
        self._subdivide()
        return [child for child in [self._nw, self._ne, self._sw, self._se] if child is not None]

    def _subdivide(self):
        # Only ever needs to be done once - do not proceed if there are children already
        if any([child is not None for child in [self._nw, self._ne, self._sw, self._se]]):
            return

        # Do not proceed if the node is terminal (includes case where node is a point)
        if self.terminal:
            return

        x_mid = int(np.floor(np.mean(self.x_lims)))
        y_mid = int(np.floor(np.mean(self.y_lims)))

        # For a single column, can only subdivide in y direction
        if self.x_lims[0] == self.x_lims[1]:
            self._nw = QuadTree((x_mid, x_mid), (self.y_lims[0], y_mid), self)
            self._sw = QuadTree((x_mid, x_mid), (y_mid + 1, self.y_lims[1]), self)
            return
        # For a single row, can only subdivide in y direction
        if self.y_lims[0] == self.y_lims[1]:
            self._nw = QuadTree((self.x_lims[0], x_mid), (y_mid, y_mid), self)
            self._ne = QuadTree((x_mid + 1, self.x_lims[1]), (y_mid, y_mid), self)
            return
        self._nw = QuadTree((self.x_lims[0], x_mid), (self.y_lims[0], y_mid), self)
        self._ne = QuadTree((x_mid + 1, self.x_lims[1]), (self.y_lims[0], y_mid), self)
        self._sw = QuadTree((self.x_lims[0], x_mid), (y_mid + 1, self.y_lims[1]), self)
        self._se = QuadTree((x_mid + 1, self.x_lims[1]), (y_mid + 1, self.y_lims[1]), self)

    def __eq__(self, other: 'QuadTree'):
        return self.x_lims == other.x_lims and self.y_lims == other.y_lims and self.parent == other.parent
