import pickle
import numpy as np
import cv2 as cv

X_train = open("/content/gdrive/My Drive/Nucleus Data/training_data.pickle","rb")
Y_train = open("/content/gdrive/My Drive/Nucleus Data/training_label.pickle","rb")

X_test = open("/content/gdrive/My Drive/Nucleus Data/testing_data.pickle","rb")
Y_test = open("/content/gdrive/My Drive/Nucleus Data/testing_labels.pickle","rb")

X_train = pickle.load(X_train)
Y_train = pickle.load(Y_train)
X_test = pickle.load(X_test)
Y_test = pickle.load(Y_test)

print(f'X_train shape: {X_train.shape}')
print(f'Y_train shape: {Y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'Y_test shape: {Y_test.shape}')


def improve_contrast_image_using_clahe(bgr_image: np.array) -> np.array:
    hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
    hsv_planes = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv.merge(hsv_planes)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

X_train_mod = []
for i in range(len(X_train)):

  X_train_mod.append(cv.resize(improve_contrast_image_using_clahe(X_train[i]), (128, 128)))

X_train = np.array(X_train_mod)

X_test_mod = []
for i in range(len(X_test)):
  X_test_mod.append(cv.resize(improve_contrast_image_using_clahe(X_test[i]), (128, 128)))

X_test = np.array(X_test_mod)
