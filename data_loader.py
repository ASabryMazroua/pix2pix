import os
from scipy import io
# from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from support_functions import relabel
import mediapipe as mp
import cv2

change_background_mp = mp.solutions.selfie_segmentation
change_bg_segment = change_background_mp.SelfieSegmentation()


class DataLoader():
    def __init__(self, img_res=(512,512), test_size=0.1, val_size=0.1):
        self.img_res = img_res[::-1]
        path = os.path.join(os.getcwd(), "dataset/annotations")
        all_files = os.listdir(path)
        self.train_files, self.test_files = train_test_split(all_files, test_size=test_size+val_size)
        self.test_files, self.val_files = train_test_split(self.test_files, test_size=val_size/(test_size+val_size))

    def load_data(self, batch_size=1, is_testing=False):
        # data_type = "train" if not is_testing else "test"
        # path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        if is_testing:
            path = self.test_files
        else:
            path = self.train_files
        batch = list(np.random.choice(path, size=batch_size))        
        imgs_A = []
        imgs_B = []
        for image in batch:
            # Read the images
            img_A = self.imread(image)
            img_A = self.resize_img(img_A)
            # Read the annotations
            img_B = self.annread(image)
            img_B = self.resize_img(img_B)
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        # data_type = "train" if not is_testing else "val"
        # path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        if is_testing:
            path = self.val_files
        else:
            path = self.train_files
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img_A = self.imread(img)
                img_B = self.annread(img)
                
                img_A = self.resize_img(img_A)
                img_B = self.resize_img(img_B)
                # h, w, _ = img.shape
                # half_w = int(w/2)
                # img_A = img[:, :half_w, :]
                # img_B = img[:, half_w:, :]

                # img_A = scipy.misc.imresize(img_A, self.img_res)
                # img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        path = os.path.join(os.getcwd(), "dataset/photos", path).replace("mat", "jpg")
        sample_img = cv2.imread(path)
        RGB_sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        result = change_bg_segment.process(RGB_sample_img)
        binary_mask = result.segmentation_mask > 0.9
        binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
        output_image = np.where(binary_mask_3, sample_img, 255)[:,:,::-1]
        # plt.imshow(output_image);plt.title("Output Image");plt.axis('off');
        # return plt.imread(path).astype(float)
        return output_image.astype(float)
    
    def annread(self, path):
        path = os.path.join(os.getcwd(), "dataset/annotations", path)
        anno = io.loadmat(path)["groundtruth"].astype(float)
        # anno = np.repeat(anno[:, :, np.newaxis], 3, axis=2)
        anno = np.array(Image.fromarray(anno.astype(np.uint8)).convert('RGB')).astype('int') 
        anno = relabel(anno) # Relabelling the segmented image
        return anno
    
    def resize_img(self, img):
        return np.array(Image.fromarray(img.astype(np.uint8)).resize(self.img_res)).astype('float32')
