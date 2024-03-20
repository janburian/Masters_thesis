import matplotlib.pyplot as plt
from tile_image import ImageSplitterMerger
import os
from extract_extracellular_matrix import color_thresholding, remove_orange_brown, make_white_background, \
    create_pink_contours, get_lobules_method_1, get_lobules_method_2
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms


def process_tile(tile: np.array) -> np.array:
    thresholded_tile = color_thresholding(tile)
    # plt.imshow(thresholded_tile[:,:,::-1])
    # plt.show()

    return thresholded_tile

def remove_extracellular_matrix(tile: np.array):
    # plt.imshow(tile)
    # plt.show()
    # lobules = get_lobules_method_1(tile, 50)
    lobules = get_lobules_method_2(tile)

    res = remove_orange_brown(tile)
    res = make_white_background(res)

    # pink_color_RGB_structures = (145, 92, 146)
    pink_color_RGB_structures = (195, 75, 182)
    pink_color_RGB_inside_lobules = (240, 231, 239)  # RGB

    pink_color_BGR = (pink_color_RGB_structures[2], pink_color_RGB_structures[1], pink_color_RGB_structures[0])
    grey_color_BGR = (pink_color_RGB_inside_lobules[2], pink_color_RGB_inside_lobules[1], pink_color_RGB_inside_lobules[0])

    res = create_pink_contours(res, lobules, pink_color_BGR, grey_color_BGR)

    # plt.imshow(res)
    # plt.show()

    return res


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
    # path_to_czi = Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'test2.czi')

    # Create an ImageSplitterMerger instance with the specified parameters
    image = ImageSplitterMerger(path_to_czi, tilesize_px=500, overlap_px=0, pixelsize_mm=[0.001, 0.001],
                                fcn=remove_extracellular_matrix)

    # Split and merge the image, applying the specified tile processing function
    merged_image = image.split_and_merge_image()

    # Display the input and merged images using Matplotlib
    plt.imshow(merged_image[:,:,::-1])
    plt.title("Merged thresholded picture")
    plt.show()

    # Save the merged image as a PNG file
    plt.imsave("output.png", merged_image[:,:,::-1])