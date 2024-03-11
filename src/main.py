import matplotlib.pyplot as plt
from tile_image import ImageSplitterMerger
import os
from extract_extracellular_matrix import color_thresholding
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms


def process_tile(tile: np.array) -> np.array:
    thresholded_tile = color_thresholding(tile)
    # plt.imshow(thresholded_tile[:,:,::-1])
    # plt.show()

    return thresholded_tile

def process_tile_test(tile: np.array) -> np.array:
    # Create a copy of the image to avoid modifying the original array
    result_tile = np.copy(tile)

    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        result_tile = to_tensor(result_tile)

    result_tile = result_tile.cpu().permute(1, 2, 0).numpy()
    result_tile = (result_tile - result_tile.min()) / (result_tile.max() - result_tile.min())  # normalize

    plt.imshow(result_tile)
    plt.show()

    # Draw the red square
    # red_color = [255, 0, 0]
    # result_tile[5:5 + 50, 5:5 + 50, :] = red_color
    return result_tile


def process_tile_test_2(tile):
    # Create a copy of the image to avoid modifying the original array
    result_tile = np.copy(tile)

    plt.imshow(result_tile)
    plt.show()

    # Draw the red square
    # red_color = [255, 0, 0]
    # result_tile[5:5 + 50, 5:5 + 50, :] = red_color
    return result_tile


if __name__ == '__main__':
    ## Define the path to the CZI file
    # path_to_czi = Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'J8_8_a.czi')
    path_to_czi = Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'J7_5_a.czi')

    # Create an ImageSplitterMerger instance with the specified parameters
    image = ImageSplitterMerger(path_to_czi, tilesize_px=1000, overlap_px=0, pixelsize_mm=[0.01, 0.01],
                                fcn=process_tile_test)

    # Split and merge the image, applying the specified tile processing function
    merged_image = image.split_and_merge_image()

    # Display the input and merged images using Matplotlib
    plt.imshow(merged_image)
    plt.title("Merged thresholded picture")
    plt.show()

    # Save the merged image as a PNG file
    plt.imsave("output.png", merged_image)