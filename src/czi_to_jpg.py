from pathlib import Path
import sys
import os

path_to_script = Path("~/GitHub/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image


def get_filenames(path: Path, accepted_extensions: list):
    """
    Returns list of filenames
    :param path: path to directory
    :param accepted_extensions:
    :return: list of filenames
    """
    filenames = [fn for fn in os.listdir(path) if fn.split(".")[-1] in accepted_extensions]

    return filenames


def czi_to_jpg_iterator(czi_directory_path: Path, pixel_size_mm: list):
    """
    Exports .czi files to .jpgs
    :param czi_directory_path: Path to .czi files directory
    :return: .jpg image
    """
    czi_files_names = get_filenames(czi_directory_path, ["czi"])

    index = 0
    while index < len(czi_files_names):
        fn_path = Path(os.path.join(Path(__file__).parent, czi_directory_path, czi_files_names[index]))
        fn_str = str(fn_path)
        if not fn_path.exists():
            break
        print(f"filename: {fn_path} {fn_path.exists()}")

        anim = scaffan.image.AnnotatedImage(path=fn_str)

        #print(anim.annotations)
        view = anim.get_full_view(
            pixelsize_mm=pixel_size_mm,
        )  # wanted pixelsize in mm in view
        # view = anim.get_view(
        #     center=(3 * 1000, 3 * 1000),
        #     pixelsize_mm=pixel_size_mm,
        #     size_mm=[1, 1]
        # )  # wanted pixelsize in mm in view
        img = view.get_raster_image()

        yield img

        index += 1
