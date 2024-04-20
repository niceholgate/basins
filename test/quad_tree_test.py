# import pytest
from src.quad_tree import QuadTree


# def _check_raises(error_type, error_message, func, *args):
#     with pytest.raises(error_type) as excinfo:
#         result = func(*args)
#     assert str(excinfo.value) == error_message


def test_iterate_boundary_coordinates_big():
    sut = QuadTree((0, 2), (0, 3))
    expected_boundary_coords = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_boundary_coordinates_2x2():
    sut = QuadTree((0, 1), (0, 1))
    expected_boundary_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_boundary_coordinates_1x2():
    sut = QuadTree((2, 2), (2, 3))
    expected_boundary_coords = [(2, 2), (2, 3)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_boundary_coordinates_1x1():
    sut = QuadTree((2, 2), (3, 3))
    expected_boundary_coords = [(2, 3)]
    assert list(sut.boundary_coordinates_generator()) == expected_boundary_coords


def test_iterate_interior_coordinates_big():
    sut = QuadTree((0, 3), (0, 4))
    expected_interior_coords = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert list(sut.interior_coordinates_generator()) == expected_interior_coords


def test_iterate_interior_coordinates_2x2():
    sut = QuadTree((0, 1), (0, 1))
    expected_interior_coords = []
    assert list(sut.interior_coordinates_generator()) == expected_interior_coords


def test_random_interior_coordinate():
    sut = QuadTree((0, 3), (0, 4))
    for _ in range(10):
        random_coordinates = sut.random_interior_coordinates()
        assert 0 < random_coordinates[0] < 3
        assert 0 < random_coordinates[1] < 4


def test_subdivide_big_1():
    sut = QuadTree((0, 2), (0, 3))
    sut.subdivide()
    assert sut.get_children() == [QuadTree((0, 1), (0, 1)), QuadTree((2, 2), (0, 1)),
                                  QuadTree((0, 1), (2, 3)), QuadTree((2, 2), (2, 3))]


def test_subdivide_big_2():
    sut = QuadTree((0, 3), (0, 2))
    sut.subdivide()
    assert sut.get_children() == [QuadTree((0, 1), (0, 1)), QuadTree((2, 3), (0, 1)),
                                  QuadTree((0, 1), (2, 2)), QuadTree((2, 3), (2, 2))]


def test_subdivide_2x2():
    sut = QuadTree((0, 1), (0, 1))
    sut.subdivide()
    assert sut.get_children() == [QuadTree((0, 0), (0, 0)), QuadTree((1, 1), (0, 0)),
                                  QuadTree((0, 0), (1, 1)), QuadTree((1, 1), (1, 1))]


def test_subdivide_1x2():
    sut = QuadTree((0, 0), (0, 1))
    sut.subdivide()
    assert sut.ne is None
    assert sut.se is None
    assert sut.get_children() == [QuadTree((0, 0), (0, 0)), QuadTree((0, 0), (1, 1))]


def test_subdivide_2x1():
    sut = QuadTree((0, 1), (0, 0))
    sut.subdivide()
    assert sut.sw is None
    assert sut.se is None
    assert sut.get_children() == [QuadTree((0, 0), (0, 0)), QuadTree((1, 1), (0, 0))]


def test_subdivide_1x1():
    sut = QuadTree((0, 0), (0, 0))
    sut.subdivide()
    assert sut.nw is None
    assert sut.ne is None
    assert sut.sw is None
    assert sut.se is None
    assert sut.get_children() == []