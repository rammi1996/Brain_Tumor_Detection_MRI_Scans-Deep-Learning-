# Data Argumentation

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import imutils
import matplotlib.pyplot as plt
from os import listdir
import cv2  


def augment_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )

    
    for filename in listdir(file_dir):
        # load the image
        image = cv2.imread(file_dir + '\\' + filename)
        # reshape the image
        image = image.reshape((1,)+image.shape)
        # prefix of the names for the generated sampels.
        save_prefix = 'aug_' + filename[:-4]
        # generate 'n_generated_samples' sample images
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                           save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break




augmented_data_path = 'D:/data/Downloads/archive (10)/brain_tumor_dataset/data_augmented_path/'

# augment data for the examples with label equal to 'yes' representing tumorous examples
augment_data(file_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\yes", n_generated_samples=6, save_to_dir=os.path.join(augmented_data_path, 'aug_yes'))

# augment data for the examples with label equal to 'no' representing non-tumorous examples
augment_data(file_dir=r"D:\data\Downloads\archive (10)\brain_tumor_dataset\no", n_generated_samples=9, save_to_dir=os.path.join(augmented_data_path, 'aug_no'))
