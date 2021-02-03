#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    LeakyReLU,
    UpSampling2D,
    Conv2D,
    Convolution2D,
    MaxPooling2D,
    Concatenate,
    GaussianNoise,
    GaussianDropout,
    Lambda,
    GlobalAveragePooling2D,
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

import h5py
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pathlib

import time

import math


# In[2]:


print("Tensorflow version: ", tf.version.VERSION)  # tf2
print("Keras version: ", tf.keras.__version__)  # 2.2.4-tf

# tf.enable_eager_execution()  # tf2
print("Is eager execution enabled: ", tf.executing_eagerly())
print("Is there a GPU available: ", tf.test.is_gpu_available())


# In[3]:


path_train_tfrecord = "fp56_train.tfrecord"
path_test_tfrecord = "fp56_test.tfrecord"


# # model save dir

# In[4]:


dir_model = "vgg_cam/"
pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)


# In[5]:


fp_dim = (56, 56, 6)


def _parse_function(example_proto):
    # Create a description of the features.
    feature_description = {
        "floorplan": tf.io.FixedLenFeature(
            fp_dim, tf.float32, default_value=tf.zeros(fp_dim, tf.float32)
        ),
        "plan_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "year": tf.io.FixedLenFeature([], tf.int64, default_value=-1),  # 0~9
        # "sido": tf.FixedLenFeature([], tf.int64, default_value=-1),
        # "norm_area": tf.FixedLenFeature([], tf.float32, default_value=0.0),
        # "num_rooms": tf.FixedLenFeature([], tf.int64, default_value=-1),
        # "num_baths": tf.FixedLenFeature([], tf.int64, default_value=-1),
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    return parsed_example["floorplan"], parsed_example["year"]


# In[6]:


def _onehot_year(fp, year):
    year_onehot = tf.one_hot(year, 10)  # 1970~4 -> 0, 2015~9 -> 9
    return (fp, year_onehot)


# In[7]:


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath, compression_type="GZIP")

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=4)

    ### preprocess the features

    # won't use it. use sparse_categorical_crossentropy instead of categorical_crossentropy.
    #     dataset = dataset.map(_onehot_year, num_parallel_calls=4)

    return dataset


# In[8]:


def VGG16_convolutions():
    if K.image_data_format() == "channels_last":
        input_shape = (fp_dim[0], fp_dim[1], fp_dim[2])
    else:
        input_shape = (fp_dim[2], fp_dim[0], fp_dim[1])

    model = Sequential()
    model.add(GaussianNoise(0.1, input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation="relu", name="conv1_1", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", name="conv1_2", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(128, (3, 3), activation="relu", name="conv2_1", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", name="conv2_2", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(256, (3, 3), activation="relu", name="conv3_1", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", name="conv3_2", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", name="conv3_3", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation="relu", name="conv4_1", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv4_2", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv4_3", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation="relu", name="conv5_1", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv5_2", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", name="conv5_3", padding="same"))
    return model


# In[9]:


num_classes = 10


def create_model():
    model = VGG16_convolutions()

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# In[10]:


callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    dir_model + "model-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.1%}.hdf5",
    # save_weights_only=True,
    verbose=1,
)


# In[11]:


callback_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)


# # run

# In[12]:


# .repeat().shuffle(4096).batch(8)

train_dataset = create_dataset(path_train_tfrecord).shuffle(1024).batch(8)
test_dataset = create_dataset(path_test_tfrecord).batch(8)

train_dataset, test_dataset


# In[13]:


model = create_model()
model.summary()


# In[14]:


path_best = dir_model + "model-15-1.36.hdf5"
epoch_best = 0  # 0 if starting from fresh

if epoch_best and os.path.exists(path_best):
    model.load_weights(path_best)
    history = model.fit(
        train_dataset,
        epochs=50,
        initial_epoch=epoch_best,
        validation_data=test_dataset,
        callbacks=[callback_checkpoint, callback_stop],
    )
else:
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=test_dataset,
        callbacks=[callback_checkpoint, callback_stop],
    )


# In[15]:


history.history


# In[16]:


df_hist = pd.DataFrame(
    history.history,
    index=range(epoch_best + 1, epoch_best + len(history.history["loss"]) + 1),
)
df_hist.index.name = "epoch"


# In[17]:


df_hist


# In[18]:


path_hist = dir_model + "history.csv"
df_hist.to_csv(path_hist, mode="a", header=not os.path.exists(path_hist))
