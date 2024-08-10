from typing import Optional, Dict, List, Tuple
import numpy as np
import numba as nb
import math
import src.config as cfg
from numba.experimental import jitclass
from numba import int32, optional, types, boolean

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT
nb.config.SHOW_HELP = True

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
    ('child_idx', int32),
    ('nw', optional(types.unicode_type)),
    ('ne', optional(types.unicode_type)),
    ('sw', optional(types.unicode_type)),
    ('se', optional(types.unicode_type)),
]
@jitclass(quadtree_spec)
class QuadTree:

    def __init__(self, x0: int, x1: int, y0: int, y1: int, parent: Optional[str]):
        self.id: str = str(np.random.randint(0, 1000000))
        self.x0: int = x0
        self.x1: int = x1
        self.y0: int = y0
        self.y1: int = y1
        self.parent: Optional[str] = parent
        self.terminal: bool = False
        if self.x0 == self.x1 and self.y0 == self.y1:
            self.terminal = True
        self.child_idx = -1
        self.nw: Optional[str] = None
        self.ne: Optional[str] = None
        self.sw: Optional[str] = None
        self.se: Optional[str] = None

    # @staticmethod
    # def floatToString(floatnumber: types.float32) -> str:
    #     stringNumber: str = ""
    #     whole: int = math.floor(floatnumber)
    #     frac: int = 0
    #     digits: float = float(floatnumber % 1)
    #     digitsTimes100: float = float(digits) * float(100.0)
    #     if digitsTimes100 is not None:
    #         frac = math.floor(digitsTimes100)
    #     stringNumber = str(whole) + "." + str(frac)
    #     return stringNumber

    def boundary_coordinates_generator(self) -> Tuple[int, int]:
        for x in range(self.x0, self.x1 + 1):
            yield x, self.y0
        for y in range(self.y0 + 1, self.y1 + 1):
            yield self.x1, y
        for x in range(self.x1-1, self.x0-1, -1):
            yield x, self.y1
        for y in range(self.y1-1, self.y0, -1):
            yield self.x0, y

    def interior_coordinates_generator(self) -> Tuple[int, int]:
        for x in range(self.x0 + 1, self.x1):
            for y in range(self.y0 + 1, self.y1):
                yield x, y

    def random_interior_x(self) -> int:
        return int(np.random.randint(self.x0 + 1, self.x1))

    def random_interior_y(self) -> int:
        return int(np.random.randint(self.y0 + 1, self.y1))

    def get_next_node_dfs(self, quadtree_dict: Dict[str, 'QuadTree']) -> str:
        # if self._next_node_dfs_generator is None:
        #     self._next_node_dfs_generator = self._create_next_node_dfs_generator(quadtree_dict)
        # return next(self._next_node_dfs_generator)
        self.child_idx += 1
        children: List[str] = self.get_children(quadtree_dict)
        if self.child_idx == len(children):
            if self.parent is None:
                return None
            return quadtree_dict[self.parent].get_next_node_dfs(quadtree_dict)
        return children[self.child_idx]

    # def _create_next_node_dfs_generator(self, quadtree_dict):
    #     for child in self.get_children(quadtree_dict):
    #         yield child
    #     while True:
    #         if self.parent is None:
    #             yield None
    #         yield quadtree_dict[self.parent].get_next_node_dfs(quadtree_dict)

    def get_children(self, quadtree_dict: Dict[str, 'QuadTree']) -> List[str]:
        self.subdivide(quadtree_dict)
        children = []
        if self.nw: children.append(self.nw)
        if self.ne: children.append(self.ne)
        if self.sw: children.append(self.sw)
        if self.se: children.append(self.se)
        return children
        # return [child for child in [self._nw, self._ne, self._sw, self._se] if child is not None]

    def subdivide(self, quadtree_dict: Dict[str, 'QuadTree']) -> None:
        # Only ever needs to be done once - do not proceed if there are children already
        if self.nw or self.ne or self.sw or self.sw:
        # if any([child is not None for child in [self._nw, self._ne, self._sw, self._se]]):
            return

        # Do not proceed if the node is terminal (includes case where node is a point)
        if self.terminal:
            return

        x_mid: int = int(np.floor((self.x0+self.x1)/2))
        y_mid: int = int(np.floor((self.y0+self.y1)/2))

        # For a single column, can only subdivide in y direction
        if self.x0 == self.x1:
            self.set_nw(QuadTree(x_mid, x_mid, self.y0, y_mid, self.id), quadtree_dict)
            self.set_sw(QuadTree(x_mid, x_mid, y_mid + 1, self.y1, self.id), quadtree_dict)
            return
        # For a single row, can only subdivide in y direction
        if self.y0 == self.y1:
            self.set_nw(QuadTree(self.x0, x_mid, y_mid, y_mid, self.id), quadtree_dict)
            self.set_ne(QuadTree(x_mid + 1, self.x1, y_mid, y_mid, self.id), quadtree_dict)
            return
        self.set_nw(QuadTree(self.x0, x_mid, self.y0, y_mid, self.id), quadtree_dict)
        self.set_ne(QuadTree(x_mid + 1, self.x1, self.y0, y_mid, self.id), quadtree_dict)
        self.set_sw(QuadTree(self.x0, x_mid, y_mid + 1, self.y1, self.id), quadtree_dict)
        self.set_se(QuadTree(x_mid + 1, self.x1, y_mid + 1, self.y1, self.id), quadtree_dict)

    def set_nw(self, qt: 'QuadTree', quadtree_dict: Dict[str, 'QuadTree']) -> None:
        quadtree_dict[qt.id] = qt
        self.nw = qt.id

    def set_ne(self, qt: 'QuadTree', quadtree_dict: Dict[str, 'QuadTree']) -> None:
        quadtree_dict[qt.id] = qt
        self.ne = qt.id

    def set_sw(self, qt: 'QuadTree', quadtree_dict: Dict[str, 'QuadTree']) -> None:
        quadtree_dict[qt.id] = qt
        self.sw = qt.id

    def set_se(self, qt: 'QuadTree', quadtree_dict: Dict[str, 'QuadTree']) -> None:
        quadtree_dict[qt.id] = qt
        self.se = qt.id

    def equals(self, other: 'QuadTree') -> bool:
        return self.x0 == other.x0 and self.x1 == other.x1 \
               and self.y0 == other.y0 and self.y1 == other.y1 \
               and self.parent == other.parent
    # def __eq__(self, other: 'QuadTree') -> bool:
    #     return self.x0 == other.x0 and self.x1 == other.x1 \
    #            and self.y0 == other.y0 and self.y1 == other.y1 \
    #            and self.parent == other.parent
