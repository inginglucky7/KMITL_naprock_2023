import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2 as cv

image_path =  "D:\Dataset\Train\Hand_Fist\Dataset_Hand_Fist_163159.jpg"

original_image = Image.open(image_path)

original_image_np = np.array(original_image)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10)),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
])

augmented_images = [seq.augment_image(original_image_np)for _ in range(5)]

plt.figure(figsize=(10, 5))
plt.subplot(1, 6, 1)
plt.imshow(original_image_np)
plt.title('Original')

for i, augmented_image in enumerate(augmented_images):
    plt.subplot(1, 6, i + 2)
    plt.imshow(augmented_image)
    plt.title(f"Augmented {i + 1}")

plt.tight_layout()
plt.show()