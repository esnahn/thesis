#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic("matplotlib", "inline")

import cv2, matplotlib
import numpy as np
from skimage.morphology import (
    skeletonize,
    skeletonize_3d,
    medial_axis,
    thin,
    local_minima,
    local_maxima,
)
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import distance_transform_edt

from math import sqrt

import matplotlib.pyplot as plt

from os.path import expanduser, splitext
from os import scandir, makedirs

# import random

import csv

from tqdm import tnrange, tqdm_notebook

from pathlib import Path

debug = False  # plot every steps


# In[2]:


from floorplan_analysis import read_bgr_from_image_unicode
from floorplan_analysis import read_from_csv

from floorplan_analysis import get_unit_mask

from floorplan_analysis import align_fp, rescale_fp
from floorplan_analysis import mono_fp
from floorplan_analysis import read_mono_from_image_unicode, save_mono_to_image_unicode


# # process files

# In[3]:


def process_floorplan_mono(
    path_from, area, filename_to, dir_to="/data/fp_img_processed/", ext_to=".png"
):
    try:
        bgr = read_bgr_from_image_unicode(path_from)
        unit_comb = get_unit_mask(bgr)
        unit_comb = rescale_fp(unit_comb, area)
        unit_comb = align_fp(unit_comb)

        mono = mono_fp(unit_comb)
        save_mono_to_image_unicode(mono, dir_to + filename_to + ext_to, ext_to)
    except:
        print(filename_to)


# In[4]:


process_floorplan_mono("/fp_img/10001_57B.jpg", 85, "mono", dir_to="")


# In[5]:


plt.imshow(read_mono_from_image_unicode("mono.png"))


# In[6]:


bgr = read_bgr_from_image_unicode("/fp_img/10001_57B.jpg")
plt.imshow(bgr)


# In[7]:


unit_comb = get_unit_mask(bgr)


# In[8]:


unit_comb = rescale_fp(unit_comb, 85)


# In[9]:


unit_comb = align_fp(unit_comb)
plt.imshow(unit_comb[:, :, 2])


# In[10]:


mono = mono_fp(unit_comb)


# # multiprocessing

# In[11]:


from multiprocessing import Pool


def worker(x, y):
    return x * y


with Pool(7) as p:
    output = p.starmap(worker, [(i, 2 * i) for i in range(101)], chunksize=100)

print(output)


# # main

# In[12]:


dir_ID_from = "/fp_img/"
dir_IDs_exclude = "/data/exclude/"

dir_from = "/fp_img/"

dir_to = "/data/fp_img_processed/"
makedirs(dir_to, exist_ok=True)

ext_to = ".png"

### all of the plans
ID_ext_dict = {
    splitext(f.name)[0]: splitext(f.name)[1]
    for f in scandir(dir_ID_from)
    if f.is_file()
}
print(len(ID_ext_dict.keys()), "floorplans")


# In[13]:


list(ID_ext_dict.items())[:10]


# In[14]:


files_IDs_exclude = list(Path(expanduser(dir_IDs_exclude)).glob("*.csv"))

# don't repeat the process
files_IDs_exclude.append(Path("processed.csv"))

print(files_IDs_exclude)


# In[15]:


IDs_excl = set()
for file_excl in files_IDs_exclude:
    _, file_excl_list = read_from_csv(str(file_excl))
    if file_excl_list:
        list_excl = [row[0] for row in file_excl_list]
    IDs_excl |= set(list_excl)
    print(file_excl, "processed:", len(list_excl), "floorplans to exclude")

# _, fp_img_processed_list = read_from_csv(exp_path_fp_img)
# if fp_img_processed_list:
#     list_excl = [row[0] for row in fp_img_processed_list]
#     IDs_excl |= set(list_excl)
#     print(len(list_excl), "floorplans already processed")


# In[16]:


import pandas as pd

path_csv = "fp_refined.csv"

df = pd.read_csv(path_csv)
df = df.set_index("id_after")
df


# In[17]:


ID_set = set(ID_ext_dict.keys()).difference(IDs_excl)
ID_set = ID_set.intersection(df.index)
IDs = list(ID_set)
print(len(IDs), "floorplans to go")


# In[18]:


paths_from = [dir_from + ID + ID_ext_dict[ID] for ID in IDs]
print(paths_from[:10])


# In[19]:


df.Area


# In[20]:


area_list = [df.Area[ID] for ID in IDs]
area_list[:10]


# In[21]:


from multiprocessing import Pool

makedirs(dir_to, exist_ok=True)

with Pool(7) as p:
    p.starmap(process_floorplan_mono, zip(paths_from, area_list, IDs))


#     ls /data/fp_img_processed/ | wc -l
