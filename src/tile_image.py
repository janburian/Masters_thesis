import numpy as np
import cv2
import os
from pathlib import Path
import tqdm
import sys
import scaffan
import scaffan.image

path_to_script = Path("~/GitHub/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))


class ImageSplitterMerger(object):
    """Class which represents splitter and merger of the image"""
    def __init__(self, img_path: Path, tilesize_px: int, overlap_px: int, pixelsize_mm: list, fcn=None):
        self.img_path = img_path
        self.tilesize_px = tilesize_px
        self.overlap_px = overlap_px
        self.pixelsize_mm = pixelsize_mm
        self.fcn = fcn

        anim = self.load_image(img_path)
        # view = anim.get_full_view(level)
        view = anim.get_full_view(pixelsize_mm=pixelsize_mm[0])
        shape = view.get_size_on_pixelsize_mm()  # returns rows and cols
        shape = np.append(shape, 3)  # Added image channel = 3

        setattr(self, "view", view)
        setattr(self, "anim", anim)
        setattr(self, "img_shape", shape)

    def get_num_cols_rows(self, img_shape):
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        num_rows = int(np.ceil(img_shape[0] / (tilesize_px - overlap_px)))
        num_cols = int(np.ceil(img_shape[1] / (tilesize_px - overlap_px)))

        return num_rows, num_cols

    @staticmethod
    def load_image(img_path: Path):
        img_path_str = str(img_path)
        print(os.path.exists(img_path_str))
        anim = scaffan.image.AnnotatedImage(path=img_path_str)

        return anim

    def get_number_tiles(self, shape):
        nx, ny = self.get_num_cols_rows(shape)
        return nx * ny

    def split_iterator(self):
        """Split image into tiles."""
        img = self.img_path
        img_shape = self.img_shape
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px

        (num_rows, num_cols) = self.get_num_cols_rows(img_shape)

        for i in range(num_rows):
            for j in range(num_cols):
                row_start = i * (tilesize_px - overlap_px)
                row_end = row_start + tilesize_px

                col_start = j * (tilesize_px - overlap_px)
                col_end = col_start + tilesize_px

                # New tile with overlap for each iteration
                overlapped_tile = self.get_tile_overlap(col_end, col_start, overlap_px, row_end,
                                                       row_start, img_shape)

                # plt.imshow(overlapped_tile)
                # plt.show()
                yield overlapped_tile

    def get_tile_overlap(self, col_end: int, col_start: int, overlap_px: int, row_end: int, row_start: int, img_shape: np.array):
        """Returns tile with the overlap."""
        pixelsize_mm = self.pixelsize_mm

        # Calculate the valid region to copy from the original image
        img_row_start = max(0, row_start - overlap_px)
        img_row_end = min(img_shape[0], row_end + overlap_px)
        img_col_start = max(0, col_start - overlap_px)
        img_col_end = min(img_shape[1], col_end + overlap_px)

        # Calculate the corresponding region in the tile with overlap
        # pad_row_start = max(0, overlap_px - (row_start - img_row_start))
        # pad_row_end = pad_row_start + img_row_end - img_row_start
        # pad_col_start = max(0, overlap_px - (col_start - img_col_start))
        # pad_col_end = pad_col_start + img_col_end - img_col_start

        # Copy the valid region from the original image to the tile with overlap
        # overlapped_tile[pad_row_start:pad_row_end, pad_col_start:pad_col_end] = img[img_row_start:img_row_end,
        #                                                                             img_col_start:img_col_end]
        view = self.anim.get_view(
            location_mm=(img_col_start * pixelsize_mm[0], img_row_start * pixelsize_mm[1]),  # Changed this line
            pixelsize_mm=pixelsize_mm,
            size_on_level=(self.tilesize_px, self.tilesize_px)
        )

        # overlapped_tile = view.get_raster_image()
        overlapped_tile = view.get_region_image(as_gray=False)
        # plt.imshow(overlapped_tile)
        # plt.show()

        return overlapped_tile

    def merge_tiles_to_image(self, tiles: list):
        """Merge tiles into image and remove overlap."""
        orig_image = self.img_path
        tilesize_px = self.tilesize_px
        overlap_px = self.overlap_px
        img_shape = self.img_shape
        (num_rows, num_cols) = self.get_num_cols_rows(img_shape)

        # Initialize the merged image
        merged_image = np.zeros(img_shape, dtype="uint8")

        with tqdm.tqdm(total=num_rows * num_cols, desc="Merging Tiles") as pbar:
            for i in range(num_rows):
                for j in range(num_cols):
                    idx = i * num_cols + j

                    row_start = i * (tilesize_px - overlap_px)
                    row_end = min(row_start + tilesize_px, merged_image.shape[0])  # Adjust for edge tiles

                    col_start = j * (tilesize_px - overlap_px)
                    col_end = min(col_start + tilesize_px, merged_image.shape[1])  # Adjust for edge tiles

                    # Remove overlap from all sides of the tile
                    tile_no_overlap = tiles[idx][overlap_px:(overlap_px + row_end - row_start),
                                                      overlap_px:(overlap_px + col_end - col_start)]

                    # plt.imshow(tile_no_overlap)
                    # plt.show()

                    # Calculate the corresponding region in the merged image
                    merged_row_start = i * (tilesize_px - overlap_px)
                    merged_row_end = merged_row_start + row_end - row_start
                    merged_col_start = j * (tilesize_px - overlap_px)
                    merged_col_end = merged_col_start + col_end - col_start

                    # Copy the tile without overlap to the merged image
                    merged_image[merged_row_start:merged_row_end, merged_col_start:merged_col_end] = tile_no_overlap
                    pbar.update(1)

        return merged_image

    def split_and_merge_image(self):
        """Split and merge image, process tile."""
        processed_tiles = []
        total_tiles = self.get_number_tiles(self.img_shape)

        for tile in tqdm.tqdm(self.split_iterator(), total=total_tiles, desc="Splitting and Processing Tiles"):
            processed_tile = np.copy(tile)
            if self.fcn is not None:
                processed_tile = self.fcn(tile)
            processed_tiles.append(processed_tile)
            # plt.figure()
            # plt.imshow(tile)
            # plt.imshow(processed_tile)
            # plt.show()

        merged_img = self.merge_tiles_to_image(processed_tiles)

        return merged_img


def load_image(img_path):
    img = cv2.imread(str(img_path))

    return img


def create_test_image():
    x, y = np.indices([300, 500])
    center1 = (256, 256)
    # radius1 = 20
    mask = (x - center1[0]) ** 2 + (y - center1[1]) ** 2

    # plt.imshow(mask)
    # plt.show()

    return mask


