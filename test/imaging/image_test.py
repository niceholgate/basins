import src.imaging.image as image

from pathlib import Path
from typing import List
import os


def test_png_to_mp4():
    test_dir = Path(__file__).parent.parent/'resources/png_frames'
    assert len(get_files_in_dir_with_extension(test_dir, '.mp4')) == 0
    image.png_to_mp4(test_dir, 5)
    assert len(get_files_in_dir_with_extension(test_dir, '.mp4')) == 1
    for mp4_file in get_files_in_dir_with_extension(test_dir, '.mp4'):
        os.remove(test_dir/mp4_file)


def test_rgb_to_mp4():
    test_dir = Path(__file__).parent.parent/'resources/rgb_frames'
    assert len(get_files_in_dir_with_extension(test_dir, '.mp4')) == 0
    image.rgb_to_mp4(test_dir, 5)
    assert len(get_files_in_dir_with_extension(test_dir, '.mp4')) == 1
    for mp4_file in get_files_in_dir_with_extension(test_dir, '.mp4'):
        os.remove(test_dir/mp4_file)


def get_files_in_dir_with_extension(dir: Path, extension: str) -> List[str]:
    return [file for file in os.listdir(dir) if file.endswith(extension)]