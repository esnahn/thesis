#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic("matplotlib", "inline")

from PIL import Image

import cv2, matplotlib
import numpy as np

from math import sqrt

import matplotlib.pyplot as plt

from os.path import expanduser, splitext
from os import scandir, makedirs

import csv

from tqdm.notebook import trange, tqdm

from pathlib import Path

debug = False  # plot every steps


# In[2]:


from floorplan_analysis import read_from_csv


# # CSV

# In[3]:


dir_from = "/data/fp_img_processed/"

csv_to = "processed.csv"

### all of the plans
ID_path_dict = {splitext(f.name)[0]: f.path for f in scandir(dir_from) if f.is_file()}
print(len(ID_path_dict.keys()), "floorplans")


# In[4]:


list(ID_path_dict.items())[:10]


# In[5]:


with open(csv_to, "w", newline="", encoding="utf-8-sig") as csvfile:
    listwriter = csv.writer(csvfile)
    listwriter.writerow(["ID"])

    IDs_error = []
    for ID, path in tqdm(ID_path_dict.items(), desc="Processing plans"):
        try:
            listwriter.writerow([ID])
        except:
            IDs_error.append(ID)
    print(len(IDs_error))
    print(IDs_error)


# # analysis

# In[6]:


import pandas as pd

df = pd.read_csv(csv_to)
# df = df.set_index("ID")


# In[7]:


df


# In[8]:


df_old = pd.read_csv("processed_0812.csv")

df_old[~df_old.index.isin(df.index)]

