from typing import Optional, List, Tuple
import numpy as np
import numba as nb
import src.config as cfg
from numba.experimental import jitclass
from numba import int32, optional, types, boolean

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT
nb.config.SHOW_HELP = True


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
        # TODO: dict of children
        self.nw: Optional['QuadTree'] = None
        self.ne: Optional['QuadTree'] = None
        self.sw: Optional['QuadTree'] = None
        self.se: Optional['QuadTree'] = None

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

    def get_children(self) -> List['QuadTree']:
        self._subdivide()
        return [child for child in [self.nw, self.ne, self.sw, self.se] if child is not None]

    def _subdivide(self) -> None:
        # Only ever needs to be done once - do not proceed if there are children already
        # (there is always a nw if there are any children)
        if self.nw is not None:
            return

        # Do not proceed if the node is terminal (includes case where node is a point)
        if self.terminal:
            return

        x_mid: int = int(np.floor((self.x0+self.x1)/2))
        y_mid: int = int(np.floor((self.y0+self.y1)/2))

        # For a single column, can only subdivide in y direction
        if self.x0 == self.x1:
            self.nw = QuadTree(x_mid, x_mid, self.y0, y_mid, self)
            self.sw = QuadTree(x_mid, x_mid, y_mid + 1, self.y1, self)
            return
        # For a single row, can only subdivide in y direction
        if self.y0 == self.y1:
            self.nw = QuadTree(self.x0, x_mid, y_mid, y_mid, self)
            self.ne = QuadTree(x_mid + 1, self.x1, y_mid, y_mid, self)
            return
        self.nw = QuadTree(self.x0, x_mid, self.y0, y_mid, self)
        self.ne = QuadTree(x_mid + 1, self.x1, self.y0, y_mid, self)
        self.sw = QuadTree(self.x0, x_mid, y_mid + 1, self.y1, self)
        self.se = QuadTree(x_mid + 1, self.x1, y_mid + 1, self.y1, self)

    def equals(self, other: 'QuadTree') -> bool:
        same_coords = self.x0 == other.x0 and self.x1 == other.x1 \
               and self.y0 == other.y0 and self.y1 == other.y1
        same_parents = (self.parent is None and other.parent is None) \
            or (self.parent is not None and other.parent is not None and self.parent.id == other.parent.id)
        return same_coords and same_parents


if cfg.ENABLE_JIT:
    node_type.define(optional(QuadTree.class_type.instance_type))

# def log_to_file_from_jit():
#     with nb.objmode():
#         f = open("logfile.txt", "a")
#         f.write('\n Hello!')
#         f.close()