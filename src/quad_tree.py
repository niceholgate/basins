from typing import Tuple, Optional, Generator
import numpy as np
import uuid
from numba.experimental import jitclass
# from numba import int32, optional, types, boolean, string

# quadtree_type = deferred_type()
# quadtree_type.define(toto.class_type.instance_type)

# spec = [('a',float64),('b',float64),('c',toto_type)]


# quadtree_spec = [
#     ('id', string),
#     ('x_lims', types.Tuple([int32, int32])),
#     ('y_lims', types.Tuple([int32, int32])),
#     ('terminal', boolean),
#     ('parent', optional(string))
# ]
# @jitclass(quadtree_spec)
class QuadTree:
    id: str
    x_lims: Tuple[int, int]
    y_lims: Tuple[int, int]
    terminal: bool = False
    parent: Optional[str] = None
    _nw: Optional[str] = None
    _ne: Optional[str] = None
    _sw: Optional[str] = None
    _se: Optional[str] = None
    _next_node_dfs_generator: Generator = None

    def __init__(self, x_lims: Tuple[int, int], y_lims: Tuple[int, int], parent: Optional[str]):
        self.id = str(uuid.uuid4())
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

    def get_next_node_dfs(self, quadtree_dict):
        if self._next_node_dfs_generator is None:
            self._next_node_dfs_generator = self._create_next_node_dfs_generator(quadtree_dict)
        return next(self._next_node_dfs_generator)

    def _create_next_node_dfs_generator(self, quadtree_dict):
        for child in self.get_children(quadtree_dict):
            yield child
        while True:
            if self.parent is None:
                yield None
            yield quadtree_dict[self.parent].get_next_node_dfs(quadtree_dict)

    def get_children(self, quadtree_dict):
        self._subdivide(quadtree_dict)
        return [child for child in [self._nw, self._ne, self._sw, self._se] if child is not None]

    def _subdivide(self, quadtree_dict):
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
            self._set_nw(QuadTree((x_mid, x_mid), (self.y_lims[0], y_mid), self.id), quadtree_dict)
            self._set_sw(QuadTree((x_mid, x_mid), (y_mid + 1, self.y_lims[1]), self.id), quadtree_dict)
            return
        # For a single row, can only subdivide in y direction
        if self.y_lims[0] == self.y_lims[1]:
            self._set_nw(QuadTree((self.x_lims[0], x_mid), (y_mid, y_mid), self.id), quadtree_dict)
            self._set_ne(QuadTree((x_mid + 1, self.x_lims[1]), (y_mid, y_mid), self.id), quadtree_dict)
            return
        self._set_nw(QuadTree((self.x_lims[0], x_mid), (self.y_lims[0], y_mid), self.id), quadtree_dict)
        self._set_ne(QuadTree((x_mid + 1, self.x_lims[1]), (self.y_lims[0], y_mid), self.id), quadtree_dict)
        self._set_sw(QuadTree((self.x_lims[0], x_mid), (y_mid + 1, self.y_lims[1]), self.id), quadtree_dict)
        self._set_se(QuadTree((x_mid + 1, self.x_lims[1]), (y_mid + 1, self.y_lims[1]), self.id), quadtree_dict)

    def _set_nw(self, qt: 'QuadTree', quadtree_dict):
        quadtree_dict[qt.id] = qt
        self._nw = qt.id

    def _set_ne(self, qt: 'QuadTree', quadtree_dict):
        quadtree_dict[qt.id] = qt
        self._ne = qt.id

    def _set_sw(self, qt: 'QuadTree', quadtree_dict):
        quadtree_dict[qt.id] = qt
        self._sw = qt.id

    def _set_se(self, qt: 'QuadTree', quadtree_dict):
        quadtree_dict[qt.id] = qt
        self._se = qt.id

    def __eq__(self, other: 'QuadTree'):
        return self.x_lims == other.x_lims and self.y_lims == other.y_lims and self.parent == other.parent
