import numpy as np
import cv2
import os
from pathlib import Path
from matplotlib import pyplot as plt


class ImageSplitterMerger(object):
    """Class which represents large image"""
    def __init__(self, path_to_image: Path, img_array: np.array, tilesize_px: int, overlap_px: int):
        self.path_to_image = path_to_image
        self.img_array = img_array
        self.tilesize_px = tilesize_px
        self.overlap_px = overlap_px

    # def get_orig_image_size(self, img_array: np.array):
    #     orig_image_size = (img_array.shape[0], img_array.shape[1])
    #
    #     return orig_image_size
    #
    # def load_image(self):
    #     img_path = self.path_to_image
    #     img_array = cv2.imread(str(img_path))
    #
    #     return img_array

    def get_num_cols_rows(self):
        img = self.img_array
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        num_rows = int(np.ceil(img.shape[0] / (tilesize_px - overlap_px)))
        num_cols = int(np.ceil(img.shape[1] / (tilesize_px - overlap_px)))

        return (num_rows, num_cols)

    def split_iterator(self):
        """Split image into tiles."""
        img = self.img_array
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        (num_rows, num_cols) = self.get_num_cols_rows()

        for i in range(num_rows):
            for j in range(num_cols):
                row_start = i * (tilesize_px - overlap_px)
                row_end = row_start + tilesize_px

                col_start = j * (tilesize_px - overlap_px)
                col_end = col_start + tilesize_px

                # Create a new padded tile for each iteration
                padded_tile = np.zeros((tilesize_px + 2 * overlap_px, tilesize_px + 2 * overlap_px), dtype=img.dtype)

                # Calculate the valid region to copy from the original image
                img_row_start = max(0, row_start - overlap_px)
                img_row_end = min(img.shape[0], row_end + overlap_px)
                img_col_start = max(0, col_start - overlap_px)
                img_col_end = min(img.shape[1], col_end + overlap_px)

                # Calculate the corresponding region in the padded tile
                pad_row_start = max(0, overlap_px - (row_start - img_row_start))
                pad_row_end = pad_row_start + img_row_end - img_row_start
                pad_col_start = max(0, overlap_px - (col_start - img_col_start))
                pad_col_end = pad_col_start + img_col_end - img_col_start

                # Copy the valid region from the original image to the padded tile
                padded_tile[pad_row_start:pad_row_end, pad_col_start:pad_col_end] = img[img_row_start:img_row_end,
                                                                                    img_col_start:img_col_end]

                tile_image_object = ImageTile(padded_tile)

                # plt.imshow(padded_tile)
                # plt.show()
                yield tile_image_object

    def merge_tiles_to_image(self, tiles: list):
        """Merge tiles into image and remove padding."""
        orig_image = self.img_array
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px
        (num_rows, num_cols) = self.get_num_cols_rows()

        # Initialize the merged image
        merged_image = np.zeros(orig_image.shape, dtype=orig_image.dtype)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j

                row_start = i * (tilesize_px - overlap_px)
                row_end = min(row_start + tilesize_px, merged_image.shape[0])  # Adjust for edge tiles

                col_start = j * (tilesize_px - overlap_px)
                col_end = min(col_start + tilesize_px, merged_image.shape[1])  # Adjust for edge tiles

                # Remove padding from all sides of the tile
                tile_no_padding = tiles[idx].tile[overlap_px:overlap_px + row_end - row_start,
                                  overlap_px:overlap_px + col_end - col_start]

                plt.imshow(tile_no_padding)
                plt.show()

                # Calculate the corresponding region in the merged image
                merged_row_start = i * (tilesize_px - overlap_px)
                merged_row_end = merged_row_start + row_end - row_start
                merged_col_start = j * (tilesize_px - overlap_px)
                merged_col_end = merged_col_start + col_end - col_start

                # Copy the tile without padding to the merged image
                merged_image[merged_row_start:merged_row_end, merged_col_start:merged_col_end] = tile_no_padding

        return merged_image

    def split_and_merge_image(self):
        processed_tiles = []
        for tile in self.split_iterator():
            processed_tile = tile.process_tile() # TODO
            processed_tile = tile
            # plt.imshow(tile.tile)
            # plt.show()
            processed_tiles.append(processed_tile)

        merged_img = self.merge_tiles_to_image(processed_tiles)

        return merged_img


class ImageTile(object):
    """Class which represents image tile."""
    def __init__(self, tile: np.array):
        self.tile = tile

    def process_tile(self):
        print("Test")

def load_image(img_path):
    img = cv2.imread(str(img_path))

    return img


def create_test_image():
    x, y = np.indices([300, 500])
    center1 = (256, 256)
    radius1 = 20
    mask = (x - center1[0]) ** 2 + (y - center1[1]) ** 2

    # plt.imshow(mask)
    # plt.show()

    return mask


path_to_img = Path(os.path.join(Path(__file__).parent.parent), 'data', 'J7_5_a.png')
img_array = load_image(path_to_img)

test_image_array = create_test_image()

image = ImageSplitterMerger(path_to_img, img_array, tilesize_px=50, overlap_px=0)
test_image = ImageSplitterMerger("", test_image_array, tilesize_px=50, overlap_px=20)

plt.imshow(test_image_array)
plt.title("Input picture")
plt.show()

merged_image = test_image.split_and_merge_image()
# imgplot = plt.imshow(merged_image[:,:,::-1])
plt.imshow(merged_image)
plt.imsave("output.png", merged_image)
plt.title("Merged picture")
plt.show()
