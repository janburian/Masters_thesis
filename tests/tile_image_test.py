import numpy as np
import pytest
from pathlib import Path
import os
from src import tile_image


def muj_processing(img: np.array) -> np.array:
    img[::2] = 0
    tile_object = tile_image.ImageTile(img)

    return tile_object

def test_image_tile():
    path_to_czi = Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'J7_5_a.czi')
    image = tile_image.ImageSplitterMerger(path_to_czi, tilesize_px=200, overlap_px=0, pixelsize_mm=[0.01, 0.01], fcn=muj_processing)
    merged_image = image.split_and_merge_image()

    assert merged_image.shape[0] > 0
    assert merged_image.shape[1] > 0
    # assert merged_image[0] == 0