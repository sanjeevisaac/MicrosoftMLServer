import os

image_location = os.path.abspath(os.path.join("C:\\workshop\\ImageSimilarity\\", "kids_girls_shoes"))

images = []
for file in [img for img in os.listdir(image_location)
if img.endswith(".tif")]:
   images.append(image_location + "\\" + file)

import pandas
image_df = pandas.DataFrame(data=dict(image=images))
print(image_df)

from microsoftml import rx_featurize, load_image, resize_image, extract_pixels, featurize_image
image_vector = rx_featurize(data=image_df, ml_transforms=[
    load_image(cols=dict(Features="image")),
    resize_image(cols="Features", width=227, height=227),
    extract_pixels(cols="Features"),
    featurize_image(cols="Features", dnn_model="Alexnet")])

print(image_vector.head())

image_location_match = os.path.abspath(os.path.join("C:\\workshop\\ImageSimilarity\\", "kids_girls_shoes_match"))
images_match = []
for file in [img for img in os.listdir(image_location_match)
if img.endswith(".tif")]:
   images_match.append(image_location_match + "\\" + file)

image_match_df = pandas.DataFrame(data=dict(image=images_match))
image_match_vectors = rx_featurize(data=image_match_df, ml_transforms=[
    load_image(cols=dict(Features="image")),
    resize_image(cols="Features", width=227, height=227),
    extract_pixels(cols="Features"),
    featurize_image(cols="Features", dnn_model="Alexnet")])
print(image_match_vectors.head())

matimg = image_vector.drop("image", axis=1).as_matrix()
matmat = image_match_vectors.drop("image", axis=1).as_matrix()

from scipy.spatial.distance import cdist
distance = cdist(matimg, matmat)

import numpy as np
idx=1
sorted = np.argsort(distance[:,idx])
print(images_match[idx])
print(images[sorted[0]])
print(images[sorted[1]])
print(images[sorted[2]])

from PIL import Image
image_tiff = Image.open(images_match[idx])
image_tiff.show() 
image_tiff = Image.open(images[sorted[0]])
image_tiff.show() 
image_tiff = Image.open(images[sorted[1]])
image_tiff.show() 
image_tiff = Image.open(images[sorted[2]])
image_tiff.show() 
