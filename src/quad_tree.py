from typing import Tuple, Optional, Generator, Dict
import numpy as np
import numba as nb
import src.config as cfg
from numba.experimental import jitclass
from numba import int32, optional, types, boolean

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT

# quadtree_type = deferred_type()
# quadtree_type.define(toto.class_type.instance_type)

# spec = [('a',float64),('b',float64),('c',toto_type)]

# TODO: numba.typed.Dict? https://numba.pydata.org/numba-doc/latest/user/jitclass.html

quadtree_spec = [
    ('id', types.unicode_type),
    ('x0', int32),
    ('x1', int32),
    ('y0', int32),
    ('y1', int32),
    ('parent', optional(types.unicode_type)),
    ('terminal', boolean),
    ('_nw', optional(types.unicode_type)),
    ('_ne', optional(types.unicode_type)),
    ('_sw', optional(types.unicode_type)),
    ('_se', optional(types.unicode_type)),
    ('_next_node_dfs_generator', optional(types.unicode_type))
]
@jitclass(quadtree_spec)
class QuadTree:

    def __init__(self, x0: int, x1: int, y0: int, y1: int, parent: Optional[str]):
        self.id: str = str(np.random.rand())
        self.x0: int = x0
        self.x1: int = x1
        self.y0: int = y0
        self.y1: int = y1
        self.parent: Optional[str] = parent
        self.terminal: bool = False
        if self.x0 == self.x1 and self.y0 == self.y1:
            self.terminal = True
        self._nw: Optional[str] = None
        self._ne: Optional[str] = None
        self._sw: Optional[str] = None
        self._se: Optional[str] = None
        self._next_node_dfs_generator: Optional[str] = None

    def boundary_coordinates_generator(self):
        for x in range(self.x0, self.x1 + 1):
            yield x, self.y0
        for y in range(self.y0 + 1, self.y1 + 1):
            yield self.x1, y
        for x in range(self.x1-1, self.x0-1, -1):
            yield x, self.y1
        for y in range(self.y1-1, self.y0, -1):
            yield self.x0, y

    def interior_coordinates_generator(self):
        for x in range(self.x0 + 1, self.x1):
            for y in range(self.y0 + 1, self.y1):
                yield x, y

    def random_interior_coordinates(self) -> (int, int):
        return int(np.random.randint(self.x0 + 1, self.x1)),\
               int(np.random.randint(self.y0 + 1, self.y1))

    def get_next_node_dfs(self, quadtree_dict: Dict[str, 'QuadTree']):
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

        x_mid = int(np.floor((self.x0+self.x1)/2))
        y_mid = int(np.floor((self.y0+self.y1)/2))

        # For a single column, can only subdivide in y direction
        if self.x0 == self.x1:
            self._set_nw(QuadTree(x_mid, x_mid, self.y0, y_mid, self.id), quadtree_dict)
            self._set_sw(QuadTree(x_mid, x_mid, y_mid + 1, self.y1, self.id), quadtree_dict)
            return
        # For a single row, can only subdivide in y direction
        if self.y0 == self.y1:
            self._set_nw(QuadTree(self.x0, x_mid, y_mid, y_mid, self.id), quadtree_dict)
            self._set_ne(QuadTree(x_mid + 1, self.x1, y_mid, y_mid, self.id), quadtree_dict)
            return
        self._set_nw(QuadTree(self.x0, x_mid, self.y0, y_mid, self.id), quadtree_dict)
        self._set_ne(QuadTree(x_mid + 1, self.x1, self.y0, y_mid, self.id), quadtree_dict)
        self._set_sw(QuadTree(self.x0, x_mid, y_mid + 1, self.y1, self.id), quadtree_dict)
        self._set_se(QuadTree(x_mid + 1, self.x1, y_mid + 1, self.y1, self.id), quadtree_dict)

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
        return self.x0 == other.x0 and self.x1 == other.x1 \
               and self.y0 == other.y0 and self.y1 == other.y1 \
               and self.parent == other.parent
