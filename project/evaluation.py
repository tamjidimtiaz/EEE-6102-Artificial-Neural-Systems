from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K
#from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
#from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K
import tensorflow as tf
from tensorflow import keras
import dill
import math
import tensorflow.python.keras.backend as K
import tensorboard
from keras.utils import plot_model
from IPython.display import Image


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

  

preds_test1 = loaded_model.predict(np.moveaxis(X_test, -1, 1), verbose=1)

# Threshold predictions
preds_test1 = np.moveaxis(preds_test1, 1, -1)
preds_test_t_1 = (preds_test1 > 0.5).astype(np.uint8)

precision = []

true_objects = len(np.unique(Y_test))
pred_objects = len(np.unique(preds_test_t_1))


intersection = np.histogram2d(Y_test.flatten(), preds_test_t_1.flatten(), bins=(true_objects, pred_objects))[0]

# Compute areas (needed for finding the union between all objects)
area_true = np.histogram(Y_test, bins = true_objects)[0]
area_pred = np.histogram(preds_test_t_1, bins = pred_objects)[0]
area_true = np.expand_dims(area_true, -1)
area_pred = np.expand_dims(area_pred, 0)

# Compute union
union = area_true + area_pred - intersection

# Exclude background from the analysis
intersection = intersection[1:,1:]
union = union[1:,1:]
union[union == 0] = 1e-9

# Compute the intersection over union
iou = (intersection / union)
print('IOU:',iou)

prec = []
IOU_image = []
for i in range(len(preds_test_t_1)):
  true_objects = len(np.unique(Y_test[i]))
  pred_objects = len(np.unique(preds_test_t_1[i]))


  intersection = np.histogram2d(Y_test[i].flatten(), preds_test_t_1[i].flatten(), bins=(true_objects, pred_objects))[0]

  # Compute areas (needed for finding the union between all objects)
  area_true = np.histogram(Y_test[i], bins = true_objects)[0]
  area_pred = np.histogram(preds_test_t_1[i], bins = pred_objects)[0]
  area_true = np.expand_dims(area_true, -1)
  area_pred = np.expand_dims(area_pred, 0)

  # Compute union
  union = area_true + area_pred - intersection

  # Exclude background from the analysis
  intersection = intersection[1:,1:]
  union = union[1:,1:]
  union[union == 0] = 1e-9

  # Compute the intersection over union
  iou = (intersection / union)
  IOU_image.append(iou)
  for t in np.arange(0.5, 1.0, 0.05):
      tp, fp, fn = precision_at(t, iou)
      p = tp / (tp + fp + fn)
      prec.append(p)

  avg_precision = np.mean(prec)

print('Avg_precision:', avg_precision)
