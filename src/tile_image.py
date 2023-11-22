import numpy as np
import cv2
import os
from pathlib import Path
from matplotlib import pyplot as plt
import tqdm


class ImageSplitterMerger(object):
    """Class which represents large image"""
    def __init__(self, path_to_image: Path, img_array: np.array, tilesize_px: int, padding_px: int):
        self.path_to_image = path_to_image
        self.img_array = img_array
        self.tilesize_px = tilesize_px
        self.padding_px = padding_px

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
        padding_px = self.padding_px

        num_rows = int(np.ceil(img.shape[0] / (tilesize_px - padding_px)))
        num_cols = int(np.ceil(img.shape[1] / (tilesize_px - padding_px)))

        return num_rows, num_cols

    def get_number_tiles(self):
        nx, ny = self.get_num_cols_rows()
        return nx * ny

    def split_iterator(self):
        """Split image into tiles."""
        img = self.img_array
        tilesize_px = self.tilesize_px
        padding_px = self.padding_px

        (num_rows, num_cols) = self.get_num_cols_rows()

        for i in range(num_rows):
            for j in range(num_cols):
                row_start = i * (tilesize_px - padding_px)
                row_end = row_start + tilesize_px

                col_start = j * (tilesize_px - padding_px)
                col_end = col_start + tilesize_px

                # New padded tile for each iteration
                dimension = img.shape[2]
                padded_tile = np.zeros((tilesize_px + 2 * padding_px, tilesize_px + 2 * padding_px, dimension), dtype="uint8")

                padded_tile = self.get_padded_tile(col_end, col_start, img, padded_tile, padding_px, row_end, row_start)
                tile_image_object = ImageTile(padded_tile)

                # plt.imshow(padded_tile)
                # plt.show()
                yield tile_image_object

    def get_padded_tile(self, col_end, col_start, img, padded_tile, padding_px, row_end, row_start):
        # Calculate the valid region to copy from the original image
        img_row_start = max(0, row_start - padding_px)
        img_row_end = min(img.shape[0], row_end + padding_px)
        img_col_start = max(0, col_start - padding_px)
        img_col_end = min(img.shape[1], col_end + padding_px)

        # Calculate the corresponding region in the padded tile
        pad_row_start = max(0, padding_px - (row_start - img_row_start))
        pad_row_end = pad_row_start + img_row_end - img_row_start
        pad_col_start = max(0, padding_px - (col_start - img_col_start))
        pad_col_end = pad_col_start + img_col_end - img_col_start

        # Copy the valid region from the original image to the padded tile
        padded_tile[pad_row_start:pad_row_end, pad_col_start:pad_col_end] = img[img_row_start:img_row_end,
                                                                            img_col_start:img_col_end]

        return padded_tile

    def merge_tiles_to_image(self, tiles: list):
        """Merge tiles into image and remove padding."""
        orig_image = self.img_array
        tilesize_px = self.tilesize_px
        padding_px = self.padding_px
        (num_rows, num_cols) = self.get_num_cols_rows()

        # Initialize the merged image
        merged_image = np.zeros(orig_image.shape, dtype="uint8")

        with tqdm.tqdm(total=num_rows * num_cols, desc="Merging Tiles") as pbar:
            for i in range(num_rows):
                for j in range(num_cols):
                    idx = i * num_cols + j

                    row_start = i * (tilesize_px - padding_px)
                    row_end = min(row_start + tilesize_px, merged_image.shape[0])  # Adjust for edge tiles

                    col_start = j * (tilesize_px - padding_px)
                    col_end = min(col_start + tilesize_px, merged_image.shape[1])  # Adjust for edge tiles

                    # Remove padding from all sides of the tile
                    tile_no_padding = tiles[idx].tile[padding_px:padding_px + row_end - row_start,
                                      padding_px:padding_px + col_end - col_start]

                    # plt.imshow(tile_no_padding)
                    # plt.show()

                    # Calculate the corresponding region in the merged image
                    merged_row_start = i * (tilesize_px - padding_px)
                    merged_row_end = merged_row_start + row_end - row_start
                    merged_col_start = j * (tilesize_px - padding_px)
                    merged_col_end = merged_col_start + col_end - col_start

                    # Copy the tile without padding to the merged image
                    merged_image[merged_row_start:merged_row_end, merged_col_start:merged_col_end] = tile_no_padding
                    pbar.update(1)

        return merged_image

    def split_and_merge_image(self):
        processed_tiles = []
        total_tiles = self.get_number_tiles()

        for tile in tqdm.tqdm(self.split_iterator(), total=total_tiles, desc="Splitting and Processing Tiles"):
            processed_tile = tile.process_tile()
            processed_tile = tile # TODO: uncomment this
            processed_tiles.append(processed_tile)

        merged_img = self.merge_tiles_to_image(processed_tiles)

        return merged_img


class ImageTile(object):
    """Class which represents image tile."""
    def __init__(self, tile: np.array):
        self.tile = tile

    def process_tile(self): # TODO: add methods
        # print("Test")
        pass


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

image = ImageSplitterMerger(path_to_img, img_array, tilesize_px=100, padding_px=10)
# test_image = ImageSplitterMerger(Path(""), test_image_array, tilesize_px=50, padding_px=20)

plt.imshow(test_image_array)
plt.title("Input picture")
plt.show()

merged_image = image.split_and_merge_image()
imgplot = plt.imshow(merged_image[:, :, ::-1])
plt.imshow(merged_image[:, :, ::-1])
# plt.imsave("output.png", merged_image)
plt.title("Merged picture")
plt.show()
