import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import os
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, closing, skeletonize, opening, binary_opening, erosion
from skimage.io import imread

def load_image(img_path: Path) -> np.array:
    img = imread(str(img_path))
    return img

def otsu_thresholding(img_array: np.array):
    # Convert image to BGR
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    grayscale_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # plt.imshow(grayscale_image[:,:,::-1], cmap='gray')
    # plt.show()
    thresh = threshold_otsu(grayscale_image)
    sample_otsu = grayscale_image > thresh
    plt.imshow(sample_otsu, cmap='gray')
    plt.show()
    # plt.imshow(np.invert(sample_otsu), cmap='gray')
    # plt.show()
    inverted_otsu_result = (np.invert(sample_otsu)).astype(np.uint8)
    # result = cv2.bitwise_and(img_array, img_array, mask=sample_otsu.astype(np.uint8))
    # plt.imshow(np.invert(result[:,:,::-1]))
    plt.imshow(inverted_otsu_result, cmap="gray")
    plt.show()
    plt.show()

    return sample_otsu

def get_lobules_method_1(img_array: np.array, threshold_value):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    ret, lobules_structure_mask = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(lobules_mask, cmap="gray")
    # plt.show()

    return lobules_structure_mask

def get_lobules_method_2(img_array: np.array):
    # Convert image to BGR
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to from RGB to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds
    lower_bound = np.array([10, 100, 50])  # Adjust these values as needed
    upper_bound = np.array([30, 255, 255])

    # Create mask
    lobules_structure_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_dilated = dilation(lobules_structure_mask)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = closing(mask_dilated)



    lobules_structure_mask = mask_dilated ^ lobules_structure_mask
    # plt.imshow(lobules_structure_mask, cmap='gray')
    # plt.show()

    return lobules_structure_mask



def basic_thresholding(img_array: np.array, threshold_value):
    # Convert image to BGR
    img_array= cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_TOZERO_INV)

    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input images
    plt.imshow(thresh1, cmap='gray')
    plt.show()
    plt.imshow(thresh2, cmap='gray')
    plt.show()
    plt.imshow(thresh3, cmap='gray')
    plt.show()
    plt.imshow(thresh4, cmap='gray')
    plt.show()
    plt.imshow(thresh5, cmap='gray')
    plt.show()

def remove_orange_brown(image):
  """
  Removes orange and brown shades from an image using HSV color space.

  Args:
      image_path (str): Path to the image file.

  Returns:
      numpy.ndarray: The modified image with reduced orange-brown shades.
  """
  # Convert image to BGR
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  # plt.imshow(image[:, :, ::-1])
  # plt.show()

  # Convert the image to HSV color space
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define lower and upper bounds for orange-brown in HSV (adjust as needed)
  # Hue ranges from 0-179, Saturation and Value range from 0-255
  lower_orange_brown = np.array([10, 50, 40])  # Orig 10, 0, 40
  upper_orange_brown = np.array([30, 200, 255])  # Orig 50, 255, 255

  # Create a mask to identify orange-brown pixels
  mask = cv2.inRange(hsv_image, lower_orange_brown, upper_orange_brown)

  # Invert the mask to target non-orange-brown pixels
  # mask = cv2.bitwise_not(mask)
  dilated_mask = dilation(mask)
  dilated_mask = dilation(dilated_mask)

  contours = dilated_mask ^ mask
  # contours = binary_closing(contours).astype("uint8")

  # plt.imshow(contours, cmap='gray')
  # plt.show()

  # Apply the mask to the original image (preserves non-orange-brown colors)
  result = cv2.bitwise_and(image, image, mask=contours)

  # plt.imshow(result)
  # plt.show()

  return result

def make_white_background(image: np.array):
    # Define black pixel value (replace with your desired black threshold if needed)
    black_pixel = (0, 0, 0)

    # Replace black pixels with white
    white_pixel = (255, 255, 255)
    mask = np.all(image == black_pixel, axis=-1)
    image[mask] = white_pixel

    return image

def create_pink_contours(image: np.array, lobules_mask: np.array, color_structures: tuple, color_inside_lobules: tuple):
    # Define white pixel value
    white_pixel = (255, 255, 255)

    # Get indices of pixels to modify based on the mask
    indices = np.where(lobules_mask == 255)

    # Change pixel color at the specified indices
    image[indices[0], indices[1], :] = color_structures

    # plt.imshow(image)
    # plt.show()

    # Create mask to identify non-white pixels
    mask = np.any(image != white_pixel, axis=-1) & np.any(image != color_structures, axis=-1)

    # Replace non-white pixels with new color
    image[mask] = color_inside_lobules

    return image

def color_thresholding(img_array: np.array):
    # Convert image to BGR
    image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds
    lower_bound = np.array([10, 100, 50])  # Adjust these values as needed
    upper_bound = np.array([30, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_dilated = dilation(mask)

    for i in range(1):
        mask_dilated = dilation(mask_dilated)

    mask = mask_dilated ^ mask
    plt.imshow(mask, cmap='gray')
    plt.show()

    # Apply mask
    result = cv2.bitwise_and(img_array, img_array, mask=mask)

    # Display result (or save using cv2.imwrite)
    plt.imshow(result)
    plt.show()

    return result

def get_contours(img_grayscale: np.array):
    img_dilated = dilation(img_grayscale)
    img_dilated = dilation(img_dilated)
    img_dilated = dilation(img_dilated)
    img_dilated = dilation(img_dilated)
    img_dilated = dilation(img_dilated)
    contours = img_dilated ^ img_grayscale
    plt.imshow(contours, cmap='gray')
    plt.show()

    return contours

def apply_mask_to_orig_image(mask, orig_image):
    result = cv2.bitwise_and(orig_image, orig_image, mask=mask.astype(np.uint8))
    # plt.imshow(result[:,:,::-1])
    # plt.show()
    return result

def get_skeleton(image: np.array):
    thresholded_image = otsu_thresholding(image)
    skeleton = skeletonize(thresholded_image)
    plt.imshow(skeleton, cmap="gray")
    plt.show()


if __name__ == '__main__':
    image_name = 'extracelluar_matrix_2.png'

    # Define the path to the  image
    path_to_image = Path(os.path.join(Path(__file__).parent.parent), 'data', image_name)
    img_array = load_image(path_to_image)
    # otsu_res = otsu_thresholding(img_array)
    # contours = get_contours(otsu_res)
    # mask_orig_image = apply_mask_to_orig_image(contours, img_array)
    # basic_thresholding(img_array, threshold_value=80)
    # res = color_thresholding(img_array)

    lobules = get_lobules_method_2(img_array)
    res = remove_orange_brown(img_array)
    res = make_white_background(res)
    pink_color_RGB_structures = (145, 92, 146)
    grey_color_RGB_inside_lobules = (240, 231, 239)  # RGB

    pink_color_BGR = (pink_color_RGB_structures[2], pink_color_RGB_structures[1], pink_color_RGB_structures[0])
    grey_color_BGR = (grey_color_RGB_inside_lobules[2], grey_color_RGB_inside_lobules[1], grey_color_RGB_inside_lobules[0])

    res = create_pink_contours(res, lobules, pink_color_BGR, grey_color_BGR)
    plt.imshow(res[:,:,::-1])
    plt.show()
