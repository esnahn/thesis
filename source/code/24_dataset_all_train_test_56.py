#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --upgrade pip
# !pip install pandas
# !cp ./fp_refined.csv /data


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


import pandas as pd

from tqdm.notebook import tnrange, tqdm

# tf.enable_eager_execution()  # tf2


# In[3]:


path_csv = "fp_refined.csv"
df = pd.read_csv(path_csv)
df


# In[4]:


df_image = pd.read_csv("processed.csv")
df_image


# In[5]:


df = df[df.id_after.isin(df_image.ID)]
df


# In[6]:


def read_mono_from_image_unicode(path):
    """workaround for non-ascii filenames"""

    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    mono = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    return mono


def fp_float_from_mono(mono):
    """create 0 or 1 binary mask image from 
    wall/entrance/LDK/bedroom/balcony/bathroom stacked array"""

    # AREA_WALL = 64
    # AREA_ENTRANCE = 32
    # AREA_LDK = 16
    # AREA_BEDROOM = 8
    # AREA_BALCONY = 4
    # AREA_BATHROOM = 2

    mask_bits = np.array([64, 32, 16, 8, 4, 2], dtype=np.uint8)
    mask = np.broadcast_to(mask_bits, (*mono.shape[:2], 6))

    unit_comb = (((np.expand_dims(mono, 2) & mask) > 0)).astype(np.float)

    return unit_comb


def pad_fp(fp, width=112, height=112):
    """place the fp at the bottom center of padded image."""
    h, w = np.subtract(fp.shape[:2], (height, width))
    if h > 0:
        fp = fp[h : h + height, :, :]
    if w > 0:
        fp = fp[:, w // 2 : w // 2 + width, :]

    h, w = np.subtract((height, width), fp.shape[:2])
    fp = np.pad(fp, ((max(h, 0), 0), (max(w // 2, 0), max(w - w // 2, 0)), (0, 0)))
    return fp


def visualize_fp(fps):
    # adjusted for different luminance
    channel_to_rgba = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # wall to black L0
            [0.0, 0.33, 0.0, 0.0],  # entrance to green L30
            [1.0, 0.25, 0.0, 0.0],  # LDK to red L57
            [0.83, 0.87, 0.0, 0.0],  # bedroom to yellow L85
            [0.0, 0.26, 1.0, 0.0],  # balcony to blue L40
            [0.0, 0.81, 0.76, 0.0],  # bathroom to cyan L75
        ]
    )

    # make colors subtractive
    channel_to_rgba[:, 0:3] -= 1

    # put it on white
    fps_rgba = np.clip(
        np.array([1.0, 1.0, 1.0, 1.0]) + (np.array(fps) @ channel_to_rgba), 0, 1
    )
    return fps_rgba


# def _fp_from_string(bytes):
#     return np.frombuffer(bytes).reshape(28,28,6)


# In[7]:


# from os import makedirs
# makedirs('vis',exist_ok=True)


# 10001_57B
# 2112_49_0
id = "2112_49_0"
mono = read_mono_from_image_unicode(f"/data/fp_img_processed/{id}.png")
fp = fp_float_from_mono(mono)
fp = pad_fp(fp)

# fig = plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(visualize_fp(fp), )
# plt.axis('off')

plt.tight_layout()
# fig.savefig(f"vis/{id}.pdf", bbox_inches="tight", pad_inches=0)


# In[8]:


def preprocess_fp(path):
    mono = read_mono_from_image_unicode(path)
    fp = fp_float_from_mono(mono)
    fp = pad_fp(fp, 56, 56)
    return fp


# In[9]:


def preprocess_year(year):
    return max(0, (year - 1970) // 5)


# In[10]:


df_tfrecord = df.loc[
    :,
    [
        "Path",
        "id_after",
        "year",
        "sido_cluster_code",
        "norm_log_area",
        "Rooms",
        "Baths",
    ],
]


# In[11]:


df.APT_ID.unique().shape


# In[12]:


np.random.seed(1106)
id_test = np.random.choice(df.APT_ID.unique(), 3000)
id_test[:100]


# In[13]:


df_tfrecord[df_tfrecord.id_after.isin(df[~df.APT_ID.isin(id_test)].id_after)]


# In[14]:


rows_all = df_tfrecord.values
rows_train = df_tfrecord[
    df_tfrecord.id_after.isin(df[~df.APT_ID.isin(id_test)].id_after)
].values
rows_test = df_tfrecord[
    df_tfrecord.id_after.isin(df[df.APT_ID.isin(id_test)].id_after)
].values


# In[15]:


# The following functions can be used to convert a value to a type compatible
# with tf.Example.
# https://www.tensorflow.org/tutorials/load_data/tf_records


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_array_feature(value):
    """Returns a float_list from a numpy array of floats / doubles."""
    # https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64s_array_feature(value):
    """Returns an int64_list from a numpy array of ints."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))


# In[16]:


# tf.train.Feature(float_list=tf.train.FloatList(value=processed.flatten()))

# _floats_array_feature(fp)


# In[17]:


def serialize_example(row):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    fp = preprocess_fp(row[0])
    year = preprocess_year(row[2])
    feature = {
        "floorplan": _floats_array_feature(fp.reshape(-1)),
        "plan_id": _bytes_feature(row[1].encode("utf-8")),
        "year": _int64_feature(year),  # 0~9
        "sido": _int64_feature(row[3]),  # 0~8
        "norm_area": _float_feature(row[4]),
        "num_rooms": _int64_feature(row[5]),  # 1~7
        "num_baths": _int64_feature(row[6]),  # 1~5
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


# In[18]:


example_proto = tf.train.Example.FromString(serialize_example(rows_train[0]))
# example_proto


# In[19]:


path_all_tfrecord = "fp56.tfrecord"
path_train_tfrecord = "fp56_train.tfrecord"
path_test_tfrecord = "fp56_test.tfrecord"

options_gzip = tf.io.TFRecordOptions(compression_type="GZIP")  # tf2

errors = []


for (path, rows) in zip(
    [path_all_tfrecord, path_train_tfrecord, path_test_tfrecord],
    [rows_all, rows_train, rows_test],
):
    with tf.io.TFRecordWriter(path=path, options=options_gzip) as writer:
        for row in tqdm(rows, desc="Processing " + path):
            try:
                serialized_example = serialize_example(row)
                writer.write(serialized_example)
            except:
                errors.append(row[1])


if errors:
    print(errors)

