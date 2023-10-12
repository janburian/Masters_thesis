import numpy as np
import cv2
import os
from pathlib import Path
from matplotlib import pyplot as plt


class Image(object):
    """Class which represents large image"""
    def __init__(self, path_to_image: Path, img_array: np.array):
        self.path_to_image = path_to_image
        self.img_array = img_array

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


class ImageTile(object):
    """Split image into tiles and after the processing each tile merge them again."""

    def __init__(self, image: np.array, tilesize_px: int, overlap_px: int):
        self.image = image
        self.tilesize_px = tilesize_px
        self.overlap_px = overlap_px

    def get_num_cols_rows(self):
        img = self.image.img_array
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        # num_rows = (img.shape[0] // tilesize_px) + 1
        # num_cols = (img.shape[1] // tilesize_px) + 1

        num_rows = int(np.ceil(img.shape[0] / (tilesize_px - overlap_px)))
        num_cols = int(np.ceil(img.shape[1] / (tilesize_px - overlap_px)))

        return (num_rows, num_cols)

    def split_iterator(self):
        """Split image into tiles."""
        img = self.image.img_array
        (num_rows, num_cols) = self.get_num_cols_rows()

        for i in range(num_rows):
            for j in range(num_cols):
                new_row_start = i * (self.tilesize_px - self.overlap_px)
                new_row_end = (i + 1) * (self.tilesize_px)

                new_col_start = j * (self.tilesize_px - self.overlap_px)
                new_col_end = (j + 1) * (self.tilesize_px)

                tile_image = img[new_row_start:new_row_end, new_col_start:new_col_end]

                yield tile_image


    def merge(self, tiles: list):
        """Merge tiles into image."""
        orig_image = self.image
        tilesize_px = self.tilesize_px

        orig_image_shape = orig_image.img_array.shape
        merged_image = np.zeros(orig_image_shape, dtype="uint8")
        (num_rows, num_cols) = self.get_num_cols_rows()

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j

                row_start = i * (tilesize_px - overlap_px)
                row_end = (i + 1) * tilesize_px

                col_start = j * (tilesize_px - overlap_px)
                col_end = (j + 1) * tilesize_px

                merged_image[row_start:row_end, col_start:col_end] = tiles[idx]

        return merged_image

    def process_tile(self, tile):
        pass

    def split_and_merge_image(self, tilesize_px, overlap_px):
        # image = Image(Path(os.path.join(Path(__file__).parent.parent), 'data', 'J7_5_a.png'))
        # image.load_image()
        it = ImageTile(image, tilesize_px, overlap_px)

        processed_tiles = []
        for tile in it.split_iterator():
            # processed_tile = self.process_tile(tile) # TODO
            processed_tile = tile
            plt.imshow(processed_tile)
            plt.show()
            processed_tiles.append(processed_tile)

        merged_img = it.merge(processed_tiles)

        return merged_img


def load_image(img_path):
    img = cv2.imread(str(img_path))

    return img


path_to_img = Path(os.path.join(Path(__file__).parent.parent), 'data', 'J7_5_a.png')
img_array = load_image(path_to_img)

image = Image(path_to_img, img_array)

image_tile = ImageTile(image, 50, 0)
tilesize_px = 50
overlap_px = 20
merged_image = image_tile.split_and_merge_image(tilesize_px, overlap_px)
imgplot = plt.imshow(merged_image[:,:,::-1])
plt.show()
