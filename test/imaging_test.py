import src.imaging as imaging

from pathlib import Path
from typing import List
import os


def test_rgb_to_mp4():
    test_dir = Path('./resources/rgb_frames')
    assert len(get_mp4_files_in_dir(test_dir)) == 0
    imaging.rgb_to_mp4(test_dir, 5)
    assert len(get_mp4_files_in_dir(test_dir)) == 1
    for mp4_file in get_mp4_files_in_dir(test_dir):
        os.remove(test_dir/mp4_file)


def test_png_to_mp4():
    test_dir = Path('./resources/png_frames')
    assert len(get_mp4_files_in_dir(test_dir)) == 0
    imaging.png_to_mp4(test_dir, 5)
    assert len(get_mp4_files_in_dir(test_dir)) == 1
    for mp4_file in get_mp4_files_in_dir(test_dir):
        os.remove(test_dir/mp4_file)


def get_mp4_files_in_dir(dir: Path) -> List[str]:
    return [file for file in os.listdir(dir) if file.endswith(".mp4")]