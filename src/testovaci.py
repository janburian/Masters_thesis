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
        (num_rows, num_cols) = self.get_num_cols_rows()

        for i in range(num_rows):
            for j in range(num_cols):
                new_row_start = i * (self.tilesize_px - self.overlap_px)
                new_row_end = (i + 1) * (self.tilesize_px - self.overlap_px) + self.overlap_px

                new_col_start = j * (self.tilesize_px - self.overlap_px)
                new_col_end = (j + 1) * (self.tilesize_px - self.overlap_px) + self.overlap_px

                tile_image = img[new_row_start:new_row_end, new_col_start:new_col_end]
                tile_image_object = ImageTile(tile_image)

                # plt.imshow(tile_image)
                # plt.show()
                yield tile_image_object

    def merge_tiles_to_image(self, tiles: list):
        """Merge tiles into image."""
        orig_image = self.img_array
        tilesize_px = self.tilesize_px

        orig_image_shape = orig_image.shape
        # merged_image = np.zeros(orig_image_shape, dtype="uint8") # TODO: change to this
        merged_image = np.zeros(orig_image_shape)
        (num_rows, num_cols) = self.get_num_cols_rows()

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j

                row_start = i * (tilesize_px - self.overlap_px)
                row_end = (i + 1) * tilesize_px - (i * self.overlap_px)

                col_start = j * (tilesize_px - self.overlap_px)
                col_end = (j + 1) * tilesize_px - (j * self.overlap_px)

                merged_image[row_start:row_end, col_start:col_end] = tiles[idx].tile
                # plt.imshow(merged_image)
                # plt.show()

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
test_image = ImageSplitterMerger("", test_image_array, tilesize_px=50, overlap_px=10)

plt.imshow(test_image_array)
plt.title("Input picture")
plt.show()

merged_image = test_image.split_and_merge_image()
# imgplot = plt.imshow(merged_image[:,:,::-1])
plt.imshow(merged_image)
plt.imsave("output.png", merged_image)
plt.title("Merged picture")
plt.show()
