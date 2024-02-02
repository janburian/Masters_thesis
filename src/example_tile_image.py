import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from tile_image import ImageSplitterMerger


def process_tile_test(tile: np.array) -> np.array:
    if tile.shape[2] != 3:
        raise ValueError("Image ndarray must have 3 channels for RGB.")

    # Create a copy of the image to avoid modifying the original array
    result_tile = np.copy(tile)

    # Set the red color (assuming RGB format)
    red_color = [255, 0, 0]

    # Draw the red square
    result_tile[50:50 + 50, 50:50 + 50, :] = red_color

    return result_tile


if __name__ == '__main__':
    path_to_czi = Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'J7_5_a.czi')
    image = ImageSplitterMerger(path_to_czi, tilesize_px=200, overlap_px=0, pixelsize_mm=[0.01, 0.01],
                                fcn=process_tile_test)
    # test_image = ImageSplitterMerger(test_image_array, tilesize_px=50, overlap_px=20)

    # plt.imshow(img_array[:, :, ::-1])
    # plt.title("Input picture")
    # plt.show()

    merged_image = image.split_and_merge_image()
    plt.imshow(merged_image)
    plt.title("Merged picture")
    plt.show()

    plt.imsave("output.png", merged_image)
