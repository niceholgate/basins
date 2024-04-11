import pytest
from src.quad_tree import QuadTree


def _check_raises(error_type, error_message, func, *args):
    with pytest.raises(error_type) as excinfo:
        result = func(*args)
    assert str(excinfo.value) == error_message


def _only_expected_coords_then_stop_iter(sut, expected_boundary_coords):
    expected_boundary_coords.reverse()
    while expected_boundary_coords:
        assert sut.iterate_boundary_coordinates() == expected_boundary_coords.pop()
    _check_raises(StopIteration, '', sut.iterate_boundary_coordinates)


def test_iterate_boundary_coordinates_big():
    sut = QuadTree((0, 2), (0, 3))
    expected_boundary_coords = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1)]
    _only_expected_coords_then_stop_iter(sut, expected_boundary_coords)


def test_iterate_boundary_coordinates_2x2():
    sut = QuadTree((0, 1), (0, 1))
    expected_boundary_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    _only_expected_coords_then_stop_iter(sut, expected_boundary_coords)


def test_iterate_boundary_coordinates_1x2():
    sut = QuadTree((2, 2), (2, 3))
    expected_boundary_coords = [(2, 2), (2, 3)]
    _only_expected_coords_then_stop_iter(sut, expected_boundary_coords)


def test_iterate_boundary_coordinates_1x1():
    sut = QuadTree((2, 2), (3, 3))
    expected_boundary_coords = [(2, 3)]
    _only_expected_coords_then_stop_iter(sut, expected_boundary_coords)
