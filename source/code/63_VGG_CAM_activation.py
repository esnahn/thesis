#!/usr/bin/env python
# coding: utf-8

# In[28]:


# import tensorflow as tf
# import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model

# from tensorflow.keras.layers import (
#     Input,
#     Dense,
#     Reshape,
#     Flatten,
#     Dropout,
#     BatchNormalization,
#     Activation,
#     ZeroPadding2D,
#     LeakyReLU,
#     UpSampling2D,
#     Conv2D,
#     Convolution2D,
#     MaxPooling2D,
#     Concatenate,
#     GaussianNoise,
#     GaussianDropout,
#     Lambda,
#     GlobalAveragePooling2D,
# )

# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.utils import to_categorical

# import h5py
# import pickle
# import csv

import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm

import numpy as np
import pandas as pd

# import os
# import pathlib
# from pathlib import Path

# import time

# import math


# In[2]:


path_all_tfrecord = "fp56.tfrecord"

# path_train_tfrecord = "fp56_train.tfrecord"
# path_test_tfrecord = "fp56_test.tfrecord"


# In[3]:


dir_model = "vgg_cam/"
path_best = dir_model + "model-17-1.17-53.3%.hdf5"
path_best


# # load

# In[4]:


df = pd.read_csv("vgg_5y_prediction.csv", index_col=0)
df


# In[5]:


predictions = df.prediction.to_numpy()
predictions


# In[6]:


year_true = df.true.to_numpy()
year_true


# In[7]:


ids = df.ID.tolist()
ids[60:70]


# # run

# In[9]:


from fp_tensorflow import _parse_pair_56, _parse_single_56
from fp_tensorflow import create_pair_56_dataset, create_single_dataset
from fp_tensorflow import VGG16_convolutions
from fp_tensorflow import create_vgg_5y_model

all_dataset = create_pair_56_dataset(path_all_tfrecord, "floorplan", "year").batch(64)

model = create_vgg_5y_model()
model.load_weights(path_best)


# In[10]:


model.summary()


# In[11]:


# Get the 512 input weights to the softmax.
class_weights = model.layers[-1].get_weights()[0]


# In[12]:


class_weights.shape


# In[13]:


class_weights.mean(), class_weights.std()


# In[26]:


pd.Series(class_weights.mean(axis=1)).plot.kde()


# In[23]:


# bias for 10 of 5-year classes
# model.layers[-1].get_weights()[1]


# # global average pool output

# In[29]:


gap_model = Model(inputs=model.input, outputs=model.layers[-2].output)


# In[30]:


gap_outputs = gap_model.predict(all_dataset, verbose=1)


# In[32]:


gap_outputs.shape, gap_outputs.mean(), gap_outputs.std()


# In[41]:


pd.Series(gap_outputs[116]).plot.kde(bw_method=0.02)


# In[42]:


df_gap = pd.DataFrame(gap_outputs)
df_gap


# In[43]:


df_gap.to_csv("vgg_5y_activation.csv.gz")

