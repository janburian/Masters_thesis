import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import os
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import dilation

def load_image(img_path: Path) -> np.array:
    img = cv2.imread(str(img_path))
    return img

def otsu_thresholding(img_array: np.array):
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
    result = cv2.bitwise_and(img_array, img_array, mask=sample_otsu.astype(np.uint8))
    plt.imshow(result[:,:,::-1])
    plt.show()

    return sample_otsu

def basic_thresholding(img_array: np.array, threshold_value):
    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # applying different thresholding
    # techniques on the input image
    # all pixels value above 120 will
    # be set to 255
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
  # Read the image
  # image = cv2.imread(str(image_path))

  # Convert image to BGR
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  # plt.imshow(image[:, :, ::-1])
  # plt.show()

  # Convert the image to HSV color space
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define lower and upper bounds for orange-brown in HSV (adjust as needed)
  # Hue ranges from 0-179, Saturation and Value range from 0-255
  lower_orange_brown = np.array([10, 50, 40]) # Orig 10, 0, 40
  upper_orange_brown = np.array([30, 200, 255]) # Orig 50, 255, 255

  # Create a mask to identify orange-brown pixels
  mask = cv2.inRange(hsv_image, lower_orange_brown, upper_orange_brown)

  # Invert the mask to target non-orange-brown pixels
  mask = cv2.bitwise_not(mask)
  dilated_mask = dilation(mask)
  dilated_mask = dilation(dilated_mask)

  contours = dilated_mask ^ mask

  # Apply the mask to the original image (preserves non-orange-brown colors)
  result = cv2.bitwise_and(image, image, mask=contours)

  # plt.imshow(result)
  # plt.show()

  return result

def color_thresholding(img_array: np.array):
    # Convert to HSV
    hsv = cv2.cvtColor(img_array[:, :, ::-1], cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds
    lower_bound = np.array([10, 40, 40])  # Adjust these values as needed
    upper_bound = np.array([20, 255, 255])

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

if __name__ == '__main__':
    image_name = 'extracelluar_matrix_5.png'

    # Define the path to the  image
    path_to_image = Path(os.path.join(Path(__file__).parent.parent), 'data', image_name)
    img_array = load_image(path_to_image)
    # otsu_res = otsu_thresholding(img_array)
    # contours = get_contours(otsu_res)
    # mask_orig_image = apply_mask_to_orig_image(contours, img_array)
    # basic_thresholding(img_array, threshold_value=80)
    # res = color_thresholding(img_array)

    res = remove_orange_brown(img_array)
    plt.imshow(res[:,:,::-1])
    plt.show()
