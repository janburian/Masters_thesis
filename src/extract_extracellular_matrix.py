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


def color_thresholding(img_array: np.array):
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds
    lower_bound = np.array([10, 100, 50])  # Adjust these values as needed
    upper_bound = np.array([30, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_dilated = dilation(mask)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)
    mask_dilated = dilation(mask_dilated)

    mask = mask_dilated ^ mask
    plt.imshow(mask, cmap='gray')
    plt.show()

    # Apply mask
    result = cv2.bitwise_and(img_array, img_array, mask=mask)

    # Display result (or save using cv2.imwrite)
    plt.imshow(result[:,:,::-1])
    plt.show()

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
    plt.imshow(result[:,:,::-1])
    plt.show()

if __name__ == '__main__':
    image_name = 'extracelluar_matrix.png'
    # image_name = 'orig_image_cell_nuclei.jpg'

    # Define the path to the  image
    path_to_image = Path(os.path.join(Path(__file__).parent.parent), 'data', image_name)
    img_array = load_image(path_to_image)
    # otsu_res = otsu_thresholding(img_array)
    # contours = get_contours(otsu_res)
    # apply_mask_to_orig_image(contours, img_array)
    # basic_thresholding(img_array, threshold_value=160)
    color_thresholding(img_array)
    # plt.imshow(img_array)
    # plt.show()








