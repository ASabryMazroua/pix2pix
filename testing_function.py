
import os
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from support_functions import relabel, predict_image, master_plot, validate_segmentation


model = tf.keras.models.load_model('saved_model/generator.h5')
path = os.path.join(os.getcwd(), "dataset/annotations")
all_files = os.listdir(path)


# This function will test a random image from the dataloader
def random_test():
    # Getting an image for testing the model
    photo_index = np.random.choice(len(all_files))
    test_label = os.path.join(os.getcwd(), "dataset/annotations", all_files[photo_index])
    original_segmented_image = loadmat(test_label)["groundtruth"].astype(float)
    original_segmented_image = np.array(Image.fromarray(original_segmented_image.astype(np.uint8)).convert('RGB')).astype('int') 
    original_segmented_image = relabel(original_segmented_image)
    test_image = os.path.join(os.getcwd(), "dataset/photos", all_files[photo_index].replace("mat", "jpg"))
    original_image = plt.imread(test_image)
    # original_segmented_image = ((imgs_B[0]+1.0)*127.5).astype(int)
    predicted_segmented_image = predict_image(original_image, model=model)
    predicted_segmented_image = validate_segmentation(predicted_segmented_image)
    print("Here's the original segmentation")
    master_plot(original_image, original_segmented_image)
    print("\nHere's our predicted segmentation")
    master_plot(original_image, predicted_segmented_image)


#     plt.imshow(((imgs_A[0]+1.0)*127.5).astype(int));
#     # Generating the mask
# pred = gan.generator.predict(imgs_A)
# # plotting the prediction
# plt.imshow(((pred[0]+1.0)*127.5).astype(int))
# # Original original segmentation
# plt.imshow(((imgs_B[0]+1.0)*127.5).astype(int));
# # 
# np.unique(((imgs_B[0]+1.0)*127.5).astype(int))
# # 
# np.unique(((pred[0]+1.0)*127.5).astype(int))


# Testing on a sample image

# # Test an image with its label
# # Choose a random number
# photo_index = 1
# # Get and plot the segemnted image
# path = os.path.join(os.getcwd(), r"dataset\annotations")
# all_files = os.listdir(path)
# test_label = os.path.join(os.getcwd(), r"dataset\annotations", all_files[photo_index])
# label_image = loadmat(test_label)["groundtruth"]
# label_image = relabel(label_image)
# plt.imshow(label_image)
# # Get and plot the original image itself
# test_image = os.path.join(os.getcwd(), r"dataset\photos", all_files[photo_index].replace("mat", "jpg"))
# original_image = plt.imread(test_image)
# plt.imshow(original_image)


# master_plot(original_image, label_image, label_names=new_labels_list)
