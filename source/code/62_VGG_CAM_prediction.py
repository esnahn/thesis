#!/usr/bin/env python
# coding: utf-8

# !pip install --upgrade pip matplotlib

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


path_all_tfrecord = "fp56.tfrecord"

# path_train_tfrecord = "fp56_train.tfrecord"
# path_test_tfrecord = "fp56_test.tfrecord"


# In[3]:


dir_model = "vgg_cam/"
path_best = dir_model + "model-17-1.17-53.3%.hdf5"
path_best


# In[4]:


from fp_tensorflow import _parse_pair_56, _parse_single_56
from fp_tensorflow import create_pair_56_dataset, create_single_dataset
from fp_tensorflow import VGG16_convolutions
from fp_tensorflow import create_vgg_5y_model

all_dataset = create_pair_56_dataset(path_all_tfrecord, "floorplan", "year").batch(64)

model = create_vgg_5y_model()
model.load_weights(path_best)


# # predict

# In[5]:


predictions = model.predict(all_dataset, verbose=1)


# In[6]:


predictions = np.argmax(predictions, 1)


# In[7]:


predictions[:128]


# In[8]:


all_year = create_single_dataset(path_all_tfrecord, "year")
year_true = np.fromiter((y.numpy() for y in all_year), int)
year_true.shape, year_true[:10]


# In[9]:


all_id = create_single_dataset(path_all_tfrecord, "plan_id")
ids = [i.numpy().decode() for i in all_id]
ids[:10]


# In[10]:


df = pd.DataFrame(
    zip(ids, year_true, predictions), columns=["ID", "true", "prediction"]
)
df


# In[11]:


df.to_csv("vgg_5y_prediction.csv")


# In[12]:


crosstab = pd.crosstab(df.true, df.prediction)
crosstab[0] = 0
crosstab = crosstab.reindex(index=range(10), columns=range(10), fill_value=0)
crosstab


# In[13]:


fig = plt.figure(figsize=(7, 5), dpi=300)
ax = fig.gca()

c = ax.pcolor(crosstab.transpose(), cmap="BuGn")

ax.set_aspect("equal")
ax.set_xlabel("True Completion Year")
ax.set_ylabel("Prediction")

ax.set_xticks(range(0, 11, 2))
ax.set_yticks(range(0, 11, 2))
ax.set_xticklabels(range(1970, 2021, 10))
ax.set_yticklabels(range(1970, 2021, 10))

fig.colorbar(c, ax=ax)

fig.savefig("vgg_5y_heatmap.pdf", bbox_inches="tight", pad_inches=0)
fig.savefig("vgg_5y_heatmap.png", bbox_inches="tight", pad_inches=0)


# ### 눈 먼 정확도는 딱 한 구간 틀린 게 많다는 걸 보여주지 못함...

# In[14]:


correct = df[df.true == df.prediction].shape[0]
total = df.shape[0]
correct, total, f"{correct/total:.2%}"


# In[15]:


one_off = (
    df[df.true - 1 == df.prediction].shape[0]
    + df[df.true + 1 == df.prediction].shape[0]
)
one_off, f"{one_off/total:.2%}", f"{(correct+one_off)/total:.2%}"
