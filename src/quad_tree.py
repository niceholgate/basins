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
    ('x_lims', nb.typeof((0, 100))),
    ('y_lims', nb.typeof((0, 100))),
    ('parent', optional(node_type)),
    ('terminal', boolean),
    ('nw', optional(node_type)),
    ('ne', optional(node_type)),
    ('sw', optional(node_type)),
    ('se', optional(node_type)),
]
@jitclass(quadtree_spec)
class QuadTree:

    def __init__(self, x_lims: Tuple[int, int], y_lims: Tuple[int, int], parent: Optional['QuadTree']):
        self.id: str = str(np.random.randint(0, 1000000))
        self.x_lims: Tuple[int, int] = x_lims
        self.y_lims: Tuple[int, int] = y_lims
        self.parent: Optional['QuadTree'] = parent
        self.terminal: bool = False
        if self.x_lims[0] == self.x_lims[1] and self.y_lims[0] == self.y_lims[1]:
            self.terminal = True
        # TODO: dict of children
        self.nw: Optional['QuadTree'] = None
        self.ne: Optional['QuadTree'] = None
        self.sw: Optional['QuadTree'] = None
        self.se: Optional['QuadTree'] = None

    def boundary_coordinates_generator(self) -> Tuple[int, int]:
        for x in range(self.x_lims[0], self.x_lims[1] + 1):
            yield x, self.y_lims[0]
        for y in range(self.y_lims[0] + 1, self.y_lims[1] + 1):
            yield self.x_lims[1], y
        for x in range(self.x_lims[1]-1, self.x_lims[0]-1, -1):
            yield x, self.y_lims[1]
        for y in range(self.y_lims[1]-1, self.y_lims[0], -1):
            yield self.x_lims[0], y

    def interior_coordinates_generator(self) -> Tuple[int, int]:
        for x in range(self.x_lims[0] + 1, self.x_lims[1]):
            for y in range(self.y_lims[0] + 1, self.y_lims[1]):
                yield x, y

    def random_interior_x(self) -> int:
        return int(np.random.randint(self.x_lims[0] + 1, self.x_lims[1]))

    def random_interior_y(self) -> int:
        return int(np.random.randint(self.y_lims[0] + 1, self.y_lims[1]))

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

        x_mid: int = int(np.floor((self.x_lims[0]+self.x_lims[1])/2))
        y_mid: int = int(np.floor((self.y_lims[0]+self.y_lims[1])/2))

        # For a single column, can only subdivide in y direction
        if self.x_lims[0] == self.x_lims[1]:
            self.nw = QuadTree((x_mid, x_mid), (self.y_lims[0], y_mid), self)
            self.sw = QuadTree((x_mid, x_mid), (y_mid + 1, self.y_lims[1]), self)
            return
        # For a single row, can only subdivide in y direction
        if self.y_lims[0] == self.y_lims[1]:
            self.nw = QuadTree((self.x_lims[0], x_mid), (y_mid, y_mid), self)
            self.ne = QuadTree((x_mid + 1, self.x_lims[1]), (y_mid, y_mid), self)
            return
        self.nw = QuadTree((self.x_lims[0], x_mid), (self.y_lims[0], y_mid), self)
        self.ne = QuadTree((x_mid + 1, self.x_lims[1]), (self.y_lims[0], y_mid), self)
        self.sw = QuadTree((self.x_lims[0], x_mid), (y_mid + 1, self.y_lims[1]), self)
        self.se = QuadTree((x_mid + 1, self.x_lims[1]), (y_mid + 1, self.y_lims[1]), self)

    def equals(self, other: 'QuadTree') -> bool:
        same_coords = self.x_lims[0] == other.x_lims[0] and self.x_lims[1] == other.x_lims[1] \
               and self.y_lims[0] == other.y_lims[0] and self.y_lims[1] == other.y_lims[1]
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