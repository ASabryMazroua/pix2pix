# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 20:02:48 2022

@author: Ahmed
"""
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
import warnings 
warnings.filterwarnings("ignore")


# Make a prediction
def predict_image(image, model=None):
    processed_image = np.array(Image.fromarray(image.astype(np.uint8)).resize((512,512))).astype('float32')
    processed_image = np.array(processed_image)/127.5 - 1.
    if processed_image.ndim == 3:
        processed_image = np.expand_dims(processed_image, 0)
    if model is None:
        model = tf.keras.models.load_model('saved_model/generator.h5')
    pred = model.predict(processed_image)
    pred = np.round((pred+1.0)*127.5).astype(int)[0]
    pred = np.array(Image.fromarray(pred.astype(np.uint8)).convert('L').resize(image.shape[:2][::-1])).astype('int')
    return pred


# Read all the labels from the file
path = os.path.join(os.getcwd(), r"label_list")
data = loadmat(path)
original_labels = list(data.items())[3][1][0]
original_labels = [label[0] for label in original_labels]

new_labels = {
    'background': ['null'],
	'accessories': ['accessories','bag','belt','bra','bracelet','earrings','glasses','gloves','hat','necklace','purse', 'ring','scarf','sunglasses','tie','wallet','watch'],
	'shirt': ['shirt','sweatshirt','jacket','blazer', 'bodysuit', 'blouse', 'cape', 'cardigan','coat','hoodie','sweater','t-shirt','top', 'vest','suit'],
	'shoes': ['shoes','boots','clogs','heels', 'loafers','pumps','sandals','sneakers','socks','wedges'] ,
	'dress': [ 'dress', 'flats', 'intimate','jumper','romper','skirt','swimwear'],
	'hair': ['hair'],
	'pants': ['jeans','leggings','panties','pants','shorts','stockings','tights'],
	'skin': ['skin']
    }

def generate_mapping(original_labels=original_labels, new_labels=new_labels):
    mapping = {}
    new_labels_list = list(new_labels.keys())
    for new_label, old_list in new_labels.items():
        for old_label in old_list:
            old_index = original_labels.index(old_label)
            new_index = new_labels_list.index(new_label)
            mapping[old_index] = new_index
    return mapping, new_labels_list

mapping, new_labels_list = generate_mapping()
           
# This function can be used in mapping the old labels to the new values
def relabel(image, mapping=mapping):
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))
    out = np.zeros_like(image)
    for key,val in zip(k,v):
        out[image==key] = val
    return out 

# This function is used to validate that the output is a correct label
# If not, it'll convert it into 0
def validate_segmentation(segmented_image, labels_list=new_labels_list):
    segmented_image[segmented_image>len(labels_list)-1]=0
    return segmented_image

# This function will plot only the masked item and return everything else balnk
def plot_item(item, image, label_image, labels=new_labels_list):
    label = labels.index(item)
    masked_image = image.copy()
    masked_image[label_image!=label] = [255,255,255]
    plt.imshow(masked_image)
    plt.title(item)
    return masked_image

# This function will plot each item in one image
def master_plot(image, segmented_image, label_names=new_labels_list):
    if segmented_image.ndim >2:
        segmented_image = np.array(Image.fromarray(segmented_image.astype(np.uint8)).convert('L')).astype('int')
    available_labels = np.unique(segmented_image)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show();
    for i in range(len(available_labels)):
        masked_image = image.copy()
        masked_image[segmented_image!=available_labels[i]] = [255,255,255]
        plt.imshow(masked_image)
        plt.title(label_names[available_labels[i]])
        plt.axis('off')
        plt.show();
        