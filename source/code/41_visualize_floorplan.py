#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --upgrade pip
# !pip install pandas
# !cp ./fp_refined.csv /data


# In[2]:


get_ipython().run_line_magic("matplotlib", "inline")

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


import pandas as pd

from tqdm.notebook import tnrange, tqdm

# tf.enable_eager_execution()  # tf2


# In[3]:


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


# In[21]:


from os import makedirs


# 10001_57B
# 2112_49_0


def visualize_from_file(ids, dir_to="vis"):

    makedirs(dir_to, exist_ok=True)

    for id in ids:
        mono = read_mono_from_image_unicode(f"/data/fp_img_processed/{id}.png")
        fp = fp_float_from_mono(mono)
        fp_rgba = visualize_fp(fp)

        fp_light = (
            cv2.cvtColor(fp_rgba.astype(np.uint8) * 255, cv2.COLOR_RGB2Lab)[:, :, 0]
            / 100
        )
        fp_rgba = pad_fp(fp_rgba, fp_light.shape[1], fp_light.shape[0])

        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(fp_rgba)
        plt.axis("off")

        plt.tight_layout()
        fig.savefig(f"{dir_to}/{id}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


visualize_from_file(["10001_57B"])


# In[22]:


from pathlib import Path
import csv


def read_from_csv(filepath, columns=False):
    if Path(filepath).is_file():

        with open(filepath, "r", newline="", encoding="utf-8-sig") as csvfile:
            listreader = csv.reader(csvfile)
            if columns:
                columns = next(listreader)
            readlist = list(listreader)

    else:
        columns = []
        readlist = []

    if columns:
        return columns, readlist
    else:
        return readlist


top_ids = read_from_csv("vgg_activation_top10.csv")
top_ids


# In[23]:


ids_flat = [id for line in top_ids for id in line]
ids_flat


# In[24]:


visualize_from_file(ids_flat)
