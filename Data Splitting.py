import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def count_augmented_images(augmented_data_path):
    augmented_yes_path = os.path.join(augmented_data_path, 'aug_yes')
    augmented_no_path = os.path.join(augmented_data_path, 'aug_no')

    num_augmented_yes_images = len(os.listdir(augmented_yes_path))
    num_augmented_no_images = len(os.listdir(augmented_no_path))

    print(f"Number of augmented 'yes' images: {num_augmented_yes_images}")
    print(f"Number of augmented 'no' images: {num_augmented_no_images}")

def data_summary(main_path):
    aug_yes_path = os.path.join(main_path, 'data_augmented_path', 'aug_yes')
    aug_no_path = os.path.join(main_path, 'data_augmented_path', 'aug_no')
    
    n_pos = len(os.listdir(aug_yes_path))
    n_neg = len(os.listdir(aug_no_path))
    m = n_pos + n_neg
    
    print(f"Total number of samples: {m}")
    print(f"Number of positive samples: {n_pos}")
    print(f"Number of negative samples: {n_neg}")

def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in os.listdir(directory):
            image = cv2.imread(os.path.join(directory, filename))
            image = crop_brain_contour(image, plot=False)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            image = image / 255.
            X.append(image)
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X, y


def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    image = image[y:y + h, x:x + w]
    return image

def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Example usage:
if __name__ == "__main__":
    main_path = r"D:\data\Downloads\archive (10)\brain_tumor_dataset"
    augmented_data_path = os.path.join(main_path, 'data_augmented_path')
    count_augmented_images(augmented_data_path)
    data_summary(main_path)
    
    aug_yes = os.path.join(augmented_data_path, 'aug_yes')
    aug_no = os.path.join(augmented_data_path, 'aug_no')
    
    image_width, image_height = 240, 240
    X, y = load_data([aug_yes, aug_no], (image_width, image_height))
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
    
    print ("number of training images = " + str(X_train.shape[0]))
    print ("number of test images = " + str(X_test.shape[0]))
    print ("number of validation images = " + str(X_val.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_val  shape: " + str(X_val.shape))
    print ("Y_val  shape: " + str(y_val.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))