# ECM removal tool
## Description
The main goal of the proposed method was to create an approach for filtering the histological image. Simply put, the input of the method consists of a histological image (WSI) in .czi format, and we want to obtain an image with the extracted intercellular mass. We are only interested in the structure of the input image. For this purpose, we tried to apply computer vision approaches like Mask R-CNN and U-Net for segmentation, and then biharmonic function and diffusion models for filtering/inpainting cell nuclei.

Due to the memory requirement of the input .czi file, this file was processed by individual square. In this tile, cell nuclei were first segmented using either Mask R-CNN (instance Segmentation) or U-Net (semantic segmentation). The segmentation output is a binary mask that contains the segmented cell nuclei (if the tile contained them). We used this mask together with the tile to filter/inpaint the cell nuclei using the biharmonic function. Finally, we performed extracellular matrix segmentation using U-Net. We saved the processed tile to the list and continued processing other tiles. After processing all the tiles, we assembled the final image from the individual tiles included in the list.

## Proposed method scheme
![alt text](https://github.com/janburian/Masters_thesis/blob/main/graphics/schema_ECM_remove_2.png)

## Obtained results
![alt text](https://github.com/janburian/Masters_thesis/blob/main/graphics/output_ECM_test.png)
![alt text](https://github.com/janburian/Masters_thesis/blob/main/graphics/output_ECM_test_2.png)
![alt text](https://github.com/janburian/Masters_thesis/blob/main/graphics/output_ECM_test_3.png)

## Demo
You can try the ECM removal tool via Google Colaboratory (https://colab.research.google.com/drive/1Ss9Z2qciXUauu5FqmUMaaBH8MZtGplRT?usp=sharing)

## Sofware for creating annotations
Zeiss ZEN (https://www.zeiss.com/microscopy/int/products/microscope-software/zen-lite.html)
Make Sense (https://www.makesense.ai/)
