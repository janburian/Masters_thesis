# Python module with the auxiliary methods for main method which target is to remove ECM

# Modules import
import requests
from gdown import download_folder
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from skimage.transform import resize
from skimage.restoration import inpaint
from skimage.morphology import dilation
import os


def download_czi_file(url_path: str, filename: str):
    url_path = os.path.join(url_path, filename)

    # Fetch the file
    response = requests.get(url_path)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a local file
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        print("Failed to fetch the file from GitHub")


def download_public_gdrive_folder(url, local_path):
    """
    Downloads a directory from a public Google Drive URL.

    Args:
        url: The public Google Drive URL of the folder.
        local_path: The local path where the folder will be downloaded.
    """
    # Extract the folder ID from the URL
    folder_id = url.split('/')[-1]
    print(folder_id)
    download_folder(id=folder_id, output=local_path, quiet=False)


@keras.saving.register_keras_serializable()
def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


@keras.saving.register_keras_serializable()
def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)  # -1 multiplied as we want to minimize this value as loss function


def do_inference_unet(img: np.array, model_unet, orig_tile_shape: tuple) -> np.array:
    # print(model_unet.input_shape[1:3])
    tensor_img = tf.convert_to_tensor(img)
    resized_image = tf.image.resize(tensor_img, model_unet.input_shape[1:3])  # Match model input size
    x = img_to_array(resized_image)
    x = x / 255.0  # Assuming model expects normalized values (0-1)
    x = np.expand_dims(x, axis=0)

    # Get segmentation mask
    mask = model_unet.predict(x)

    mask = np.squeeze(mask, axis=0)
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    # Thresholding output mask
    threshold = 0.4  # You can adjust this value based on your needs
    mask = (mask[..., 0] > threshold).astype(np.uint8)  # Assuming channel 0 and converting to uint8 for binary mask
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    mask_resized_back = resize(mask, orig_tile_shape[0:2], preserve_range=True)
    return mask_resized_back


def do_inference_mask_rcnn(image: np.array, predictor):
    # Use the detector to do inference
    result = predictor(image)
    # print(result)

    instances = result["instances"]
    mask = instances.get("pred_masks")
    mask_numpy = mask.cpu().numpy()
    mask_final = np.sum(mask_numpy.astype(np.uint8), axis=0)

    return mask_final


def apply_dilation(mask, num_steps_dilation) -> np.array:
    for i in range(num_steps_dilation):
        mask = dilation(mask)

    return mask


def do_inpainting_biharmonic(orig_tile: np.array, mask: np.array) -> np.array:
    img_removed_nuclei = inpaint.inpaint_biharmonic(orig_tile, mask, channel_axis=-1)
    # plt.imshow(img_removed_nuclei[:,:,::-1])
    # plt.show()

    return img_removed_nuclei


def remove_ECM_unet(img: np.array, model_unet_ECM, orig_tile_shape):
    tensor_img = tf.convert_to_tensor(img)
    resized_image = tf.image.resize(tensor_img, model_unet_ECM.input_shape[1:3])  # Match model input size
    x = img_to_array(resized_image)
    # print("Test: " + str(x.shape))
    x = np.expand_dims(x, axis=0)

    # Get segmentation mask
    mask = model_unet_ECM.predict(x)
    mask = np.squeeze(mask, axis=0)

    mask_resized_back = resize(mask, orig_tile_shape[0:2], preserve_range=True)
    # print(mask_resized_back)

    return mask_resized_back


def process_tile_unet(tile: np.array, model_unet, model_unet_ECM) -> np.array:
    orig_tile_shape = tile.shape
    mask = do_inference_unet(tile, model_unet, orig_tile_shape)
    # mask = create_mask_from_inferences(res_inference)
    # print(type(mask))
    if np.any(mask):  # determine whether in mask are white pixels
        mask_dil = apply_dilation(mask, 5)
        # plt.imshow(mask_dil)
        # plt.show()
        img_removed_cell_nuclei = do_inpainting_biharmonic(tile, mask_dil)
        img_removed_extracellular_matrix = remove_ECM_unet(img_removed_cell_nuclei, model_unet_ECM, orig_tile_shape)
        # plt.imshow(img_removed_extracellular_matrix)
        # plt.show()
        # print(img_removed_extracellular_matrix)
        # return img_removed_cell_nuclei
        # img_removed_extracellular_matrix_new_color = create_pink_contours(img_removed_extracellular_matrix)
        # img_removed_extracellular_matrix = convert_grayscale_to_RGB(img_removed_extracellular_matrix)
        # plt.imshow(img_removed_extracellular_matrix_new_color)
        # plt.show()
        # print(img_removed_extracellular_matrix_new_color)

        # fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        # axes[0].imshow(tile)
        # axes[0].set_title("Original Tile")
        # axes[0].axis('off')
        # axes[1].imshow(mask_dil, cmap="gray")
        # axes[1].set_title("Mask")
        # axes[1].axis('off')
        # axes[2].imshow(img_removed_cell_nuclei)
        # axes[2].set_title("Tile with removed cell nuclei")
        # axes[2].axis('off')
        # axes[3].imshow(img_removed_extracellular_matrix)
        # axes[3].set_title("Removed ECM")
        # axes[3].axis('off')
        # plt.show()

        # plt.imshow(img_removed_extracellular_matrix)
        # plt.show()
        # print(img_removed_extracellular_matrix.shape)

        return img_removed_extracellular_matrix

        # return np.stack((img_removed_extracellular_matrix,) * 3, axis=-1) # TODO

    else:  # no mask in tile; returning the original tile
        # plt.imshow(tile)
        # plt.show()
        # print(tile)
        # return tile
        tile_normalized = tile / 255.0
        img_removed_extracellular_matrix = remove_ECM_unet(tile_normalized, model_unet_ECM, orig_tile_shape)
        # img_removed_extracellular_matrix_new_color = create_pink_contours(img_removed_extracellular_matrix)
        # img_removed_extracellular_matrix_new_color = convert_grayscale_to_RGB(img_removed_extracellular_matrix)

        # plt.imshow(img_removed_extracellular_matrix)
        # plt.show()
        # print('TEST: ' + str(img_removed_extracellular_matrix.dtype))
        return img_removed_extracellular_matrix

        # return np.stack((img_removed_extracellular_matrix,) * 3, axis=-1) # TODO


def process_tile_mask_rcnn(tile: np.array, predictor, model_unet_ECM) -> np.array:
    orig_tile_shape = tile.shape
    mask = do_inference_mask_rcnn(tile, predictor)
    if np.any(mask):  # determine whether in mask are white pixels
        mask_dil = apply_dilation(mask, 5)
        # plt.imshow(mask_dil)
        # plt.show()
        img_removed_cell_nuclei = do_inpainting_biharmonic(tile, mask_dil)
        img_removed_extracellular_matrix = remove_ECM_unet(img_removed_cell_nuclei, model_unet_ECM, orig_tile_shape)
        # print(img_removed_cell_nuclei.dtype)

        # print(tile_removed_cell_nuclei)
        # fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        # axes[0].imshow(tile)
        # axes[0].set_title("Original Tile")
        # axes[0].axis('off')
        # axes[1].imshow(mask_dil, cmap="gray")
        # axes[1].set_title("Mask")
        # axes[1].axis('off')
        # axes[2].imshow(img_removed_cell_nuclei)
        # axes[2].set_title("Tile with removed cell nuclei")
        # axes[2].axis('off')
        # axes[3].imshow(img_removed_extracellular_matrix)
        # axes[3].set_title("Removed ECM")
        # axes[3].axis('off')
        # plt.show()

        # print(img_removed_cell_nuclei)
        return img_removed_extracellular_matrix

    else:  # no mask in tile; returning the original tile
        # plt.imshow(tile)
        # plt.show()
        tile_normalized = tile / 255.0
        # plt.imshow(remove_ECM_unet(tile_normalized, model_unet_ECM, orig_tile_shape))
        # plt.show()
        return remove_ECM_unet(tile_normalized, model_unet_ECM, orig_tile_shape)
