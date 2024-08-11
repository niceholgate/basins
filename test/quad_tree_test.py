# import pytest
from typing import Optional
from src.quad_tree import QuadTree
from typing import List


# def _check_raises(error_type, error_message, func, *args):
#     with pytest.raises(error_type) as excinfo:
#         result = func(*args)
#     assert str(excinfo.value) == error_message


def test_iterate_boundary_coordinates_big():
    sut = QuadTree((0, 2), (0, 3), None)
    expected_boundary_coords = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_boundary_coordinates_2x2():
    sut = QuadTree((0, 1), (0, 1), None)
    expected_boundary_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_boundary_coordinates_1x2():
    sut = QuadTree((2, 2), (2, 3), None)
    expected_boundary_coords = [(2, 2), (2, 3)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_boundary_coordinates_1x1():
    sut = QuadTree((2, 2), (3, 3), None)
    expected_boundary_coords = [(2, 3)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_interior_coordinates_big():
    sut = QuadTree((0, 3), (0, 4), None)
    expected_interior_coords = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert list(sut.interior_coordinates_generator()) == expected_interior_coords


def test_iterate_interior_coordinates_2x2():
    sut = QuadTree((0, 1), (0, 1), None)
    expected_interior_coords = []
    assert list(sut.interior_coordinates_generator()) == expected_interior_coords


def test_random_interior_coordinate():
    sut = QuadTree((0, 3), (0, 4), None)
    for _ in range(10):
        assert 0 < sut.random_interior_x() < 3
        assert 0 < sut.random_interior_y() < 4


# def test_get_next_node_dfs_no_early_termination():
#     # 4x4 grid should create 1+4+16=21 QuadTrees
#     top_qt = QuadTree(0, 3, 0, 3, None)
#     n_qts = 1
#
#     next_node_dfs: Optional[QuadTree] = top_qt.get_next_node_dfs()
#     while next_node_dfs:
#         next_node_dfs = next_node_dfs.get_next_node_dfs()
#         n_qts += 1
#
#     assert n_qts == 21
#
#
# def test_get_next_node_dfs_one_quadrant_early_termination():
#     # 4x4 grid with one quadrant terminal should create 1+4+12=17 QuadTrees
#     top_qt = QuadTree(0, 3, 0, 3, None)
#     n_qts = 1
#
#     next_node_dfs: Optional[QuadTree] = top_qt.get_next_node_dfs()
#     top_qt.get_children()[-1].terminal = True
#     while next_node_dfs:
#         next_node_dfs = next_node_dfs.get_next_node_dfs()
#         n_qts += 1
#
#     assert n_qts == 17


def test_equals():
    # Different parent, same coords
    qt0 = QuadTree((0, 3), (0, 2), None)
    qt1 = QuadTree((0, 2), (0, 1), None)
    qt01 = QuadTree((0, 3), (0, 2), qt0)
    qt11 = QuadTree((0, 3), (0, 2), qt1)
    assert not qt01.equals(qt11)

    # Same parent, different coords
    qt02 = QuadTree((0, 1), (0, 2), qt0)
    assert not qt01.equals(qt02)

    # Same both
    qt02_2 = QuadTree((0, 1), (0, 2), qt0)
    assert qt02.equals(qt02_2)


def test_get_children_big_1():
    sut = QuadTree((0, 2), (0, 3), None)
    expected_children = [QuadTree((0, 1), (0, 1), sut), QuadTree((2, 2), (0, 1), sut),
                         QuadTree((0, 1), (2, 3), sut), QuadTree((2, 2), (2, 3), sut)]
    compare_expected_and_actual_children(sut, expected_children)


def test_get_children_big_2():
    sut = QuadTree((0, 3), (0, 2), None)
    expected_children = [QuadTree((0, 1), (0, 1), sut), QuadTree((2, 3), (0, 1), sut),
                         QuadTree((0, 1), (2, 2), sut), QuadTree((2, 3), (2, 2), sut)]
    compare_expected_and_actual_children(sut, expected_children)


def test_get_children_2x2():
    sut = QuadTree((0, 1), (0, 1), None)
    expected_children = [QuadTree((0, 0), (0, 0), sut), QuadTree((1, 1), (0, 0), sut),
                         QuadTree((0, 0), (1, 1), sut), QuadTree((1, 1), (1, 1), sut)]
    compare_expected_and_actual_children(sut, expected_children)


def test_get_children_1x2():
    sut = QuadTree((0, 0), (0, 1), None)
    expected_children = [QuadTree((0, 0), (0, 0), sut), QuadTree((0, 0), (1, 1), sut)]
    compare_expected_and_actual_children(sut, expected_children)


def test_get_children_2x1():
    sut = QuadTree((0, 1), (0, 0), None)
    expected_children = [QuadTree((0, 0), (0, 0), sut), QuadTree((1, 1), (0, 0), sut)]
    compare_expected_and_actual_children(sut, expected_children)


def test_get_children_1x1():
    sut = QuadTree((0, 0), (0, 0), None)
    expected_children = []
    compare_expected_and_actual_children(sut, expected_children)


def compare_expected_and_actual_children(sut: QuadTree, expected_children: List[QuadTree]):
    for actual, expected in zip(sut.get_children(), expected_children):
        assert actual.equals(expected)
