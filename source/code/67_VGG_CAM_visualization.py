#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


path_all_tfrecord = "fp56.tfrecord"


# In[3]:


dir_from = "/data/fp_img_processed/"


# In[4]:


dir_model = "vgg_cam/"
path_best = dir_model + "model-17-1.17-53.3%.hdf5"
path_best


# # model

# In[5]:


from fp_tensorflow import create_pair_56_dataset, create_single_dataset
from fp_tensorflow import create_vgg_5y_model

model = create_vgg_5y_model()
model.load_weights(path_best)
model.summary()


# # class activation map

# In[6]:


# Get the 512 input weights to the softmax.
class_weights = model.layers[-1].get_weights()[0]


# In[7]:


class_weights.shape


# In[8]:


class_weights.mean(), class_weights.std()


# In[9]:


def get_fp_output(fp, model=model):
    final_conv_layer = model.get_layer("conv5_3")
    get_output = K.function(
        [model.layers[0].input], [final_conv_layer.output, model.layers[-1].output]
    )

    conv_output, prediction = get_output(np.expand_dims(fp, 0))
    return np.squeeze(conv_output, axis=0), np.argmax(prediction)


# In[10]:


def get_fp_cam(fp, model=model):
    class_weights = model.layers[-1].get_weights()[0]

    conv_output, prediction = get_fp_output(fp, model)
    true_class_weights = class_weights[:, prediction]

    cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for i, w in enumerate(true_class_weights):
        cam += w * conv_output[:, :, i]
    return cam


# # biclust CAM

# In[11]:


biclusts = np.loadtxt("biclust_col.txt", int)
biclusts


# In[12]:


def get_biclust_cam(fp, biclust, model=model, labels=biclusts):
    conv_output, _ = get_fp_output(fp, model)

    return conv_output[..., biclusts == biclust].sum(axis=2)


# # plot

# In[13]:


def plot_bgr(img):
    fig = plt.figure(figsize=(2, 2), dpi=300)
    plt.axes().axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()


# In[14]:


def plot_rgb(img):
    fig = plt.figure(figsize=(2, 2), dpi=300)
    plt.axes().axis("off")
    plt.imshow(img)
    plt.tight_layout()


# In[15]:


def plot_gray(img, cmap=plt.cm.gray):
    fig = plt.figure(figsize=(2, 2), dpi=300)
    plt.axes().axis("off")
    plt.imshow(img, cmap=cmap)
    plt.tight_layout()


# # run

# In[16]:


from floorplan_analysis import read_mono_from_image_unicode
from floorplan_analysis import fp_float_from_mono
from floorplan_analysis import pad_fp


# In[17]:


mono = read_mono_from_image_unicode(dir_from + "2888_118A" + ".png")
fp_full = fp_float_from_mono(mono)
fp = pad_fp(fp_full, 56, 56)

conv_output, prediction = get_fp_output(fp)


# In[18]:


fp_full.shape


# In[19]:


conv_output.shape, prediction.shape


# In[20]:


prediction


# In[21]:


cam = get_fp_cam(fp)
cam = cv2.resize(cam, (56, 56))
cam /= cam.max()
cam[cam <= 0] = 0

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
# heatmap[cam < 0.2] = 0
plot_bgr(heatmap)


# In[22]:


cam = get_biclust_cam(fp, 3)

cam = cv2.resize(cam, (56, 56))
print(cam.max())
cam /= cam.max()
cam[cam <= 0] = 0

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
# heatmap[cam < 0.4] = 0
plot_bgr(heatmap)


# In[23]:


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
    return fps_rgba.astype(np.float32)


# In[24]:


rgba = visualize_fp(fp_full)
plot_rgb(rgba)


# In[25]:


def visualize_fp_cam(fp):
    fp_rgba = visualize_fp(fp)
    fp_light = cv2.cvtColor(fp_rgba, cv2.COLOR_RGB2Lab)[:, :, 0] / 100

    fp_pad = pad_fp(fp, 56, 56)

    cam = get_fp_cam(fp_pad)
    cam = cv2.resize(cam, (56, 56))
    cam /= cam.max()
    cam[cam <= 0] = 0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
    heatmap = pad_fp(heatmap, fp_light.shape[1], fp_light.shape[0])
    heatmap[fp_light == 0] = 0
    heatmap = heatmap.astype(np.float32) / 255

    return 0.7 * heatmap + 0.3 * np.expand_dims(fp_light, 2)


# In[26]:


def visualize_biclust_cam(fp, biclust):
    fp_rgba = visualize_fp(pad_fp(fp, max(56, fp.shape[1]), max(56, fp.shape[0])))
    fp_light = cv2.cvtColor(fp_rgba, cv2.COLOR_RGB2Lab)[:, :, 0] / 100

    fp_pad = pad_fp(fp, 56, 56)

    cam = get_biclust_cam(fp_pad, biclust)
    cam = cv2.resize(cam, (56, 56))
    cam /= cam.max()
    cam[cam <= 0] = 0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
    #     heatmap = pad_fp(heatmap, fp_light.shape[1], fp_light.shape[0])
    heatmap = pad_fp(heatmap, max(56, fp_light.shape[1]), max(56, fp_light.shape[0]))
    heatmap = heatmap.astype(np.float32) / 255

    return 0.7 * heatmap + 0.3 * np.expand_dims(fp_light, 2)


# In[27]:


plot_bgr(visualize_fp_cam(fp_full))


# In[28]:


plot_bgr(visualize_biclust_cam(fp_full, 3))


# # process representative floorplans

# In[29]:


df = pd.read_csv("biclust.csv")
df["area_group"] = pd.cut(df.Area, [0, 50, 60, 85, np.inf], labels=False)
df


# In[30]:


df_sample = df.groupby(["cluster", "area_group"]).sample(frac=0.005, random_state=1106)
df_sample = df_sample.sort_values(["cluster", "area_group", "year"])
df_sample


# In[31]:


pd.crosstab(df_sample.cluster, df_sample.area_group)


# In[32]:


pd.crosstab(df_sample.cluster, df_sample.area_group).max(axis=0)


# In[33]:


widths = np.asarray([3, 4, 8, 7])

coords_col = np.insert(np.cumsum(widths), 0, 0)[:-1]
coords_col


# In[34]:


heights = np.maximum(
    np.ceil(
        pd.crosstab(df_sample.cluster, df_sample.area_group).to_numpy() / widths
    ).astype(int),
    1,
).max(axis=1)
heights


# In[35]:


coords_row = np.insert(np.cumsum(heights), 0, 0)[:-1]
coords_row


# In[36]:


sum(heights)


# In[37]:


sum(widths)


# 총 31줄, 19열

# In[38]:


u = 84  # unit size
flip = False


# In[39]:


if not flip:
    img_size = (sum(heights) * u, sum(widths) * u)
else:
    img_size = (sum(widths) * u, sum(heights) * u)


# In[40]:


img_size


# In[41]:


img = np.ones(img_size + (3,), np.float32)
# img = np.zeros(img_size + (3,), np.float32)


# In[42]:


plot_bgr(pad_fp(visualize_biclust_cam(fp_full, 3), u, u, 1))


# In[43]:


df_sample[(df_sample.cluster == 0) & (df_sample.area_group == 2)]


# In[44]:


df_sample[(df_sample.cluster == 0) & (df_sample.area_group == 2)].ID.iloc[1]


# In[45]:


for ir, rr in enumerate(coords_row):
    for ic, cc in enumerate(coords_col):
        df_clust = df_sample[(df_sample.cluster == ir) & (df_sample.area_group == ic)]
        for i in range(len(df_clust)):
            r = i // widths[ic]
            c = i - r * widths[ic]
            id_ = df_clust.iloc[i].ID
            clust = df_clust.iloc[i].cluster

            img[
                (rr + r) * u : (rr + r + 1) * u, (cc + c) * u : (cc + c + 1) * u
            ] = pad_fp(
                visualize_biclust_cam(
                    fp_float_from_mono(
                        read_mono_from_image_unicode(dir_from + id_ + ".png")
                    ),
                    clust,
                ),
                u,
                u,
                1,
            )


# In[46]:


fig = plt.figure(figsize=(11, 13), dpi=300)
ax = fig.gca()
im = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_xticks((coords_col + widths / 2) * u)
ax.set_xticklabels(
    [
        "One-room\n($\leq50\mathrm{m^2}$)",
        "Small\n($\leq60\mathrm{m^2}$)",
        "Medium\n($\leq85\mathrm{m^2}$)",
        "Large\n($>85\mathrm{m^2}$)",
    ]
)
ax.set_yticks((coords_row + heights / 2 + 1 / 6) * u)
ax.set_yticklabels(range(1, biclusts.max() + 2))

ax.vlines(coords_col * u, 0, heights.sum() * u - 1, colors="k", lw=0.3)
ax.hlines(coords_row * u, 0, widths.sum() * u - 1, colors="k", lw=0.3)

fig.savefig("bam.png", bbox_inches="tight", pad_inches=0)
fig.savefig("bam.pdf", bbox_inches="tight", pad_inches=0)


# In[47]:


df_sample[(df_sample.cluster == 0)]


# 101160_113E
# 103915_112C
# 104127_107B
# 107903_113G
# 108838_117B

# In[48]:


def plot_bgr_scale(img):
    size_x, size_y = img.shape[:2]
    fig = plt.figure(figsize=(2 * size_x / 112, 2 * size_y / 112), dpi=300)
    plt.axes().axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()


# In[49]:


ir, ic, i = 15, 0, 0
u_single = 84  # 56 84 112
df_clust = df_sample[(df_sample.cluster == ir) & (df_sample.area_group == ic)]
id_ = df_clust.iloc[i].ID
print(id_)
clust = df_clust.iloc[i].cluster
plot_bgr_scale(
    pad_fp(
        visualize_biclust_cam(
            fp_float_from_mono(read_mono_from_image_unicode(dir_from + id_ + ".png")),
            clust,
        ),
        u_single,
        u_single,
        1,
    )
)


# In[53]:


def plot_bams(id_, types):
    print(id_)
    fp = fp_float_from_mono(read_mono_from_image_unicode(dir_from + id_ + ".png"))
    size_y, size_x = np.fmax(fp.shape[:2], 56)

    clust_name = [
        "8090-1",
        "8090-2",
        "8090-3",
        "9000-1",
        "9000-2",
        "9000-3",
        "9000-4",
        "00-1",
        "00-2",
        "00-3",
        "00-4",
        "0010-1",
        "0010-2",
        "0010-3",
        "10-1",
        "10-2",
    ]
    clusts = [type - 1 for type in types]

    fig, axs = plt.subplots(1, len(clusts), figsize=(11 / 4 * len(clusts), 5), dpi=300)

    for i, clust in enumerate(clusts):
        ax = axs[i]
        img = pad_fp(visualize_biclust_cam(fp, clust), size_x, size_y, 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")

        # title on bottom
        # ax.set_title(clust_name[clust], y=-size_x / 56 / 10)

        # title on top
        ax.set_title(clust_name[clust], y=1)

    plt.tight_layout()
    return fig


# # 1980년대: 판상형, 복도형

# In[54]:


df_sample[
    (df_sample.year >= 1980)
    & (df_sample.year < 1990)
    & (df_sample.cluster.isin([0, 1, 2]))
    & (df_sample.area_group == 2)
]


# 복도형 중형
#
# 서울 구로 주공1차, 73.08㎡, 1986년

# In[55]:


fig = plot_bams("137_96", [1, 2, 3])
# fig.savefig("bam.png", bbox_inches="tight", pad_inches=0)
# fig.savefig("bam.pdf", bbox_inches="tight", pad_inches=0)


# In[56]:


df_sample[
    (df_sample.year >= 1980)
    & (df_sample.year < 1990)
    & (df_sample.cluster.isin([0, 1, 2]))
    & (df_sample.area_group == 3)
]


# 복도형 대형
#
# 서울 압구정 한양7차, 106.22㎡, 1981년

# In[57]:


fig = plot_bams("501_114A", [1, 2, 3])
# fig.savefig("bam.png", bbox_inches="tight", pad_inches=0)
# fig.savefig("bam.pdf", bbox_inches="tight", pad_inches=0)


# # 1990년대: 판상형, 계단실형

# In[58]:


df_sample[(df_sample.cluster.isin([3])) & (df_sample.area_group == 2)]


# 3LDK 중형
#
# 천안 일성3차, 84.82㎡, 1994년

# In[59]:


id_ = "7479_106"

fig1 = plot_bams(id_, [1, 2, 3])
fig2 = plot_bams(id_, range(4, 7 + 1))

# fig.savefig("bam.png", bbox_inches="tight", pad_inches=0)
# fig.savefig("bam.pdf", bbox_inches="tight", pad_inches=0)


# In[60]:


df_sample[
    (df_sample.year >= 1990)
    & (df_sample.year < 2000)
    & (df_sample.cluster.isin(range(7 + 1)))
    & (df_sample.Rooms == 4)
]


# 4LDK 대형
#
# 인천 연수 하나2차, 99.42㎡, 1994년

# In[61]:


id_ = "2292_116"

fig1 = plot_bams(id_, [1, 2, 3])
fig2 = plot_bams(id_, range(4, 7 + 1))

# fig.savefig("bam.png", bbox_inches="tight", pad_inches=0)
# fig.savefig("bam.pdf", bbox_inches="tight", pad_inches=0)


# # 2000년대: 발코니, 코어 후면 배치

# In[62]:


df_sample[
    (df_sample.year >= 2000)
    & (df_sample.year < 2010)
    & (df_sample.cluster.isin([9]))
    & (df_sample.area_group == 2)
]


# 판상형
# 중형
#
# 경기
# 동화옥시죤5차,
# 84.58㎡,
# 2005년

# In[63]:


id_ = "17566_118"

fig1 = plot_bams(id_, range(8, 11 + 1))
fig2 = plot_bams(id_, range(12, 14 + 1))


# # 2010년대: 탑상형, 원룸형

# In[64]:


df_sample[(df_sample.cluster.isin([13])) & (df_sample.area_group == 2)]


# 탑상형 중앙부
# 중형
#
# 서울 서초포레스타5,
# 84.4㎡,
# 2014년

# In[65]:


id_ = "107903_112B3"

fig1 = plot_bams(id_, range(12, 14 + 1))
fig2 = plot_bams(id_, range(15, 16 + 1))


# In[66]:


df_sample[(df_sample.cluster.isin([15])) & (df_sample.area_group == 2)]


# 탑상형 단부 중형
#
# 천안 백석더샵,
# 84.25㎡,
# 2016년

# In[67]:


id_ = "108523_111C"

fig1 = plot_bams(id_, range(12, 14 + 1))
fig2 = plot_bams(id_, range(15, 16 + 1))


# In[68]:


df_sample[
    (df_sample.cluster.isin([15]))
    & (df_sample.year >= 2010)
    & (df_sample.area_group == 2)
]


# 혼합형
# (L자형 주동 계단실형 코어)
#
# 세종 가락4단지이지더원,
# 79.59㎡,
# 2014년

# In[69]:


id_ = "107076_106C"

fig1 = plot_bams(id_, range(12, 14 + 1))
fig2 = plot_bams(id_, range(15, 16 + 1))


# In[70]:


df[(df.cluster.isin([14]))].Area.mean()


# In[71]:


df[(df.cluster.isin([14]))].Area.median()


# In[72]:


df[(df.cluster.isin([14])) & (df.year >= 2010) & (df.Area >= 23) & (df.Area <= 29)]


# 원룸형 도시형생활주택
#
# 서울 역삼대명벨리온,
# 23.62㎡,
# 2012년

# In[73]:


id_ = "104259_36G"

fig = plot_bams(id_, [3, 6, 15])

