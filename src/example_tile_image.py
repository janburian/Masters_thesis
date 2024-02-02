import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from tile_image import ImageSplitterMerger  # Assuming this module is defined elsewhere


def process_tile_test(tile: np.array) -> np.array:
    """
    Process a tile by drawing a red square on it.

    Parameters:
    - tile (np.array): Input tile image (assumed to be in RGB format).

    Returns:
    - np.array: Processed tile with a red square drawn on it.
    """
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
    # Define the path to the CZI file
    path_to_czi = Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'J7_5_a.czi')

    # Create an ImageSplitterMerger instance with the specified parameters
    image = ImageSplitterMerger(path_to_czi, tilesize_px=200, overlap_px=0, pixelsize_mm=[0.01, 0.01],
                                fcn=process_tile_test)

    # Split and merge the image, applying the specified tile processing function
    merged_image = image.split_and_merge_image()

    # Display the input and merged images using Matplotlib
    plt.imshow(merged_image)
    plt.title("Merged picture")
    plt.show()

    # Save the merged image as a PNG file
    plt.imsave("output.png", merged_image)