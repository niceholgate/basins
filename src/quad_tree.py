from typing import Optional, List, Tuple
import numpy as np
import numba as nb
import src.config as cfg
from numba.experimental import jitclass
from numba import int32, optional, types, boolean

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT
nb.config.SHOW_HELP = True

# quadtree_type = deferred_type()
# quadtree_type.define(toto.class_type.instance_type)

# spec = [('a',float64),('b',float64),('c',toto_type)]

# TODO: numba.typed.Dict? https://numba.pydata.org/numba-doc/latest/user/jitclass.html
node_type = nb.deferred_type()
quadtree_spec = [
    ('id', types.unicode_type),
    ('x0', int32),
    ('x1', int32),
    ('y0', int32),
    ('y1', int32),
    ('parent', optional(node_type)),
    ('terminal', boolean),
    ('child_idx', int32),
    ('nw', optional(node_type)),
    ('ne', optional(node_type)),
    ('sw', optional(node_type)),
    ('se', optional(node_type)),
]
@jitclass(quadtree_spec)
class QuadTree:

    def __init__(self, x0: int, x1: int, y0: int, y1: int, parent: Optional['QuadTree']):
        self.id: str = str(np.random.randint(0, 1000000))
        self.x0: int = x0
        self.x1: int = x1
        self.y0: int = y0
        self.y1: int = y1
        self.parent: Optional['QuadTree'] = parent
        self.terminal: bool = False
        if self.x0 == self.x1 and self.y0 == self.y1:
            self.terminal = True
        self.child_idx = -1
        self.nw: Optional['QuadTree'] = None
        self.ne: Optional['QuadTree'] = None
        self.sw: Optional['QuadTree'] = None
        self.se: Optional['QuadTree'] = None

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

    def get_next_node_dfs(self) -> Optional['QuadTree']:
        # if self._next_node_dfs_generator is None:
        #     self._next_node_dfs_generator = self._create_next_node_dfs_generator(quadtree_dict)
        # return next(self._next_node_dfs_generator)
        self.child_idx += 1
        children: List['QuadTree'] = self.get_children()
        if self.child_idx == len(children):
            if self.parent is None:
                return None
            next_node: Optional['QuadTree'] = self.parent.get_next_node_dfs()
            if next_node is not None: return next_node
            return None
        return children[self.child_idx]

    # def _create_next_node_dfs_generator(self, quadtree_dict):
    #     for child in self.get_children(quadtree_dict):
    #         yield child
    #     while True:
    #         if self.parent is None:
    #             yield None
    #         yield quadtree_dict[self.parent].get_next_node_dfs(quadtree_dict)

    def get_children(self) -> List['QuadTree']:
        self.subdivide()
        children = []
        if self.nw is not None: children.append(self.nw)
        if self.ne is not None: children.append(self.ne)
        if self.sw is not None: children.append(self.sw)
        if self.se is not None: children.append(self.se)
        return children
        # return [child for child in [self._nw, self._ne, self._sw, self._se] if child is not None]

    def subdivide(self) -> None:
        # Only ever needs to be done once - do not proceed if there are children already
        if self.nw is not None or self.ne is not None or self.sw is not None or self.sw is not None:
        # if any([child is not None for child in [self._nw, self._ne, self._sw, self._se]]):
            return

        # Do not proceed if the node is terminal (includes case where node is a point)
        if self.terminal:
            return

        x_mid: int = int(np.floor((self.x0+self.x1)/2))
        y_mid: int = int(np.floor((self.y0+self.y1)/2))

        # For a single column, can only subdivide in y direction
        if self.x0 == self.x1:
            self.set_nw(QuadTree(x_mid, x_mid, self.y0, y_mid, self))
            self.set_sw(QuadTree(x_mid, x_mid, y_mid + 1, self.y1, self))
            return
        # For a single row, can only subdivide in y direction
        if self.y0 == self.y1:
            self.set_nw(QuadTree(self.x0, x_mid, y_mid, y_mid, self))
            self.set_ne(QuadTree(x_mid + 1, self.x1, y_mid, y_mid, self))
            return
        self.set_nw(QuadTree(self.x0, x_mid, self.y0, y_mid, self))
        self.set_ne(QuadTree(x_mid + 1, self.x1, self.y0, y_mid, self))
        self.set_sw(QuadTree(self.x0, x_mid, y_mid + 1, self.y1, self))
        self.set_se(QuadTree(x_mid + 1, self.x1, y_mid + 1, self.y1, self))

    def set_nw(self, qt: 'QuadTree') -> None:
        self.nw = qt

    def set_ne(self, qt: 'QuadTree') -> None:
        self.ne = qt

    def set_sw(self, qt: 'QuadTree') -> None:
        self.sw = qt

    def set_se(self, qt: 'QuadTree') -> None:
        self.se = qt

    def equals(self, other: 'QuadTree') -> bool:
        return self.x0 == other.x0 and self.x1 == other.x1 \
               and self.y0 == other.y0 and self.y1 == other.y1 \
               and self.parent == other.parent
    # def __eq__(self, other: 'QuadTree') -> bool:
    #     return self.x0 == other.x0 and self.x1 == other.x1 \
    #            and self.y0 == other.y0 and self.y1 == other.y1 \
    #            and self.parent == other.parent
node_type.define(QuadTree.class_type.instance_type)