import scaffan
import cv2
import os
from pathlib import Path
import sys

path_to_script = Path("~/GitHub/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image

def split_czi_image(input_path, output_folder, tile_size=(512, 512)):
    # Load CZI image using scaffan
    anim = scaffan.image.AnnotatedImage(path=input_path)

    slide = scaffan.Slide()
    # Get image dimensions
    image_width, image_height = slide.get_dimensions()

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over tiles and save them
    for x in range(0, image_width, tile_size[0]):
        for y in range(0, image_height, tile_size[1]):
            # Read tile from CZI image
            tile = slide.get_region(x, y, tile_size[0], tile_size[1])

            # Convert to OpenCV format
            tile_cv2 = tile.get_opencv_image()

            # Save tile as image file
            output_path = os.path.join(output_folder, f"tile_{x}_{y}.png")
            cv2.imwrite(output_path, tile_cv2)

if __name__ == "__main__":
    input_czi_path = str(Path(os.path.join(Path(__file__).parent.parent), 'data_czi', 'J7_5_a.czi'))
    output_folder_path = str(Path(os.path.join(Path(__file__).parent.parent), 'data', 'res.png'))

    split_czi_image(input_czi_path, output_folder_path)