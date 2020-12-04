# 부록 {.unnumbered}

## Source code {.unnumbered}

\scriptsize

10_naver_floorplan_analysis.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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
from scipy.ndimage import distance_transform_edt

from math import sqrt

import matplotlib.pyplot as plt

from os.path import expanduser, splitext
from os import scandir, makedirs

# import random

import csv

from tqdm import tnrange, tqdm_notebook

from pathlib import Path

debug = True  # plot every steps

# In[2]:

def read_from_csv(filepath):
    if Path(filepath).is_file():

        with open(filepath, "r", newline="", encoding="utf-8-sig") as csvfile:
            listreader = csv.reader(csvfile)
            columns = next(listreader)
            readlist = list(listreader)

    else:
        columns = []
        readlist = []

    return columns, readlist

def read_bgr_from_image_unicode(path):
    """workaround for non-ascii filenames"""

    stream = open(path, "rb")
    bytes_ = bytearray(stream.read())
    numpyarray = np.asarray(bytes_, dtype=np.uint8)
    bgr = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    return bgr

def save_bgr_to_image_unicode(bgr, path, ext_to=".png"):
    """workaround for non-ascii filenames"""

    _, numpyarray = cv2.imencode(ext_to, bgr)
    with open(path, "wb") as file:
        file.write(numpyarray)

# # unit mask

# In[3]:

def color_dict_mask(
    img_dict={
        "Lab": np.zeros((1, 1, 3), dtype="uint8"),
        "HSV": np.zeros((1, 1, 3), dtype="uint8"),
    },
    colors={
        "colorname": {
            "Lab": ([0, 0, 0], [255, 255, 255]),
            "HSV": ([0, 0, 0], [255, 255, 255]),
        }
    },
):
    # get masks matching any of the colors matching all descriptions

    mask = np.zeros_like(list(img_dict.values())[0][:, :, 0])
    for color_dict in colors.values():
        mask_color = np.ones_like(mask) * 255
        for colorspace, limits in color_dict.items():
            mask_colorspace = cv2.inRange(
                img_dict[colorspace], np.array(limits[0]), np.array(limits[1])
            )
            mask_color = cv2.bitwise_and(mask_color, mask_colorspace)

        mask = cv2.bitwise_or(mask, mask_color)

    return mask

def get_color_mask(
    blur={
        "Lab": np.zeros((1, 1, 3), dtype="uint8"),
        "HSV": np.zeros((1, 1, 3), dtype="uint8"),
    },
    colors={
        "colorname": {
            "Lab": ([0, 0, 0], [255, 255, 255]),
            "HSV": ([0, 0, 0], [255, 255, 255]),
        }
    },
):
    #     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    #     blur = {}
    #     blur["Lab"] = cv2.bilateralFilter(lab, 15, 25, 150)
    #     blur["BGR"] = cv2.cvtColor(blur["Lab"], cv2.COLOR_Lab2BGR)
    #     blur["HSV"] = cv2.cvtColor(blur["BGR"], cv2.COLOR_BGR2HSV)

    # get masks matching any of the colors matching all descriptions

    mask = color_dict_mask(blur, colors)

    # fill holes and remove noise

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    holes = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0]
    cv2.drawContours(mask, holes, -1, 255, -1)

    kernel_5c = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    kernel_9c = np.zeros((9, 9), np.uint8)
    cv2.circle(kernel_9c, (4, 4), 4, 1, -1)

    kernel_15c = np.zeros((15, 15), np.uint8)
    cv2.circle(kernel_15c, (7, 7), 7, 1, -1)

    # mask = cv2.erode(mask, kernel_5c, iterations=1)

    smallbits = [
        contours[i]
        for i in range(len(contours))
        if hierarchy[0][i][3] == -1 and cv2.contourArea(contours[i]) <= 100
    ]
    cv2.drawContours(mask, smallbits, -1, 0, -1)

    # removing imperfections

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        if cv2.contourArea(c) >= 100:
            mask_single_c = np.zeros_like(mask)
            cv2.drawContours(mask_single_c, c, -1, 255, -1)

            mask_single_c = cv2.morphologyEx(
                mask_single_c, cv2.MORPH_CLOSE, kernel_9c, iterations=1
            )
            mask |= mask_single_c

    return mask

def get_marked_contours(contours, marker_mask, min_marked_area):
    marked_contours = []

    for c in contours:
        mask_single_c = np.zeros_like(marker_mask)
        cv2.drawContours(mask_single_c, [c], -1, 255, -1)

        c_area = cv2.countNonZero(mask_single_c)
        marked_area = cv2.countNonZero(mask_single_c & marker_mask)

        if marked_area >= min_marked_area:
            marked_contours.append(c)

    return marked_contours

def get_marked_mask(boundary_mask, marker_mask, min_marked_area):
    contours, hierarchy = cv2.findContours(
        boundary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    marked_contours = get_marked_contours(contours, marker_mask, min_marked_area)

    marked_mask = np.zeros_like(boundary_mask)

    if marked_contours:
        cv2.drawContours(marked_mask, marked_contours, -1, 255, -1)

    return marked_mask

def get_wall_mask(bgr=np.zeros((1, 1, 3), dtype="uint8")):

    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_5c = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    # get mask based on color and shape

    redimg = bgr[:, :, 2]
    _, threshold_img_inv = cv2.threshold(redimg, 140, 255, cv2.THRESH_BINARY_INV)
    #     plt.imshow(threshold_img_inv)

    threshold_blur = cv2.medianBlur(threshold_img_inv, 5)
    #     plt.imshow(threshold_blur)
    erosion = cv2.erode(threshold_blur, kernel_3)
    opening = cv2.morphologyEx(threshold_blur, cv2.MORPH_OPEN, kernel_3)
    #     dilation = cv2.dilate(opening, kernel_3)
    #     plt.imshow(opening)
    mask = cv2.bitwise_and(threshold_img_inv, opening)
    #     plt.figure()
    #     plt.imshow(mask)

    kernel = kernel_5c

    ret, markers = cv2.connectedComponents(mask)
    #     plt.figure()
    #     plt.imshow(markers)

    wall_mask = np.zeros_like(mask)
    for i in range(1, ret):
        if (markers == i).sum() > 300:
            wall_mask |= (markers == i).astype(np.uint8) * 255
    #     plt.figure()
    #     plt.imshow(wall_mask)

    return wall_mask

def get_LDK_mask(
    blur={
        "Lab": np.zeros((1, 1, 3), dtype="uint8"),
        "HSV": np.zeros((1, 1, 3), dtype="uint8"),
    },
):
    floor_colors = {
        "floor_light": {
            "Lab": ([180, 130, 160], [220, 150, 190]),
            "HSV": ([0, 65, 180], [20, 255, 255]),
        },
        "floor_dark": {
            "Lab": ([120, 130, 150], [180, 155, 190]),
            "HSV": ([0, 90, 100], [20, 255, 230]),
        },
        "floor_watermark": {
            "Lab": ([220, 125, 145], [240, 145, 165]),
            "HSV": ([0, 65, 220], [20, 255, 255]),
        },
    }

    mask = get_color_mask(blur, floor_colors)

    return mask

def get_bedroom_mask(
    blur={
        "Lab": np.zeros((1, 1, 3), dtype="uint8"),
        "HSV": np.zeros((1, 1, 3), dtype="uint8"),
    },
):
    bedroom_boundary = {
        "bedroom_boundary": {
            "Lab": ([180, 120, 132], [254, 135, 165]),
            "HSV": ([10, 25, 200], [30, 110, 255]),
        }
    }
    bedroom_dark = {
        "bedroom_dark": {
            "Lab": ([160, 124, 139], [250, 130, 165]),
            "HSV": ([10, 30, 200], [30, 90, 250]),
        }
    }
    balcony_colors = {"balcony": {"Lab": ([240, 125, 130], [254, 135, 140])}}

    bedroom_boundary_mask = get_color_mask(blur, bedroom_boundary)
    bedroom_dark_mask = get_color_mask(blur, bedroom_dark)
    balcony_mask = get_color_mask(blur, balcony_colors)

    # remove balconies which is similarily colored

    mask_bedroom_only = np.zeros_like(bedroom_boundary_mask)

    contours, _ = cv2.findContours(
        bedroom_boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for c in contours:
        mask_single_c = np.zeros_like(mask_bedroom_only)
        cv2.drawContours(mask_single_c, [c], -1, 255, -1)

        c_area = cv2.countNonZero(mask_single_c)
        dark_area = cv2.countNonZero(mask_single_c & bedroom_dark_mask)
        balcony_area = cv2.countNonZero(mask_single_c & balcony_mask)

        if dark_area >= 1000:
            mask_bedroom_only |= mask_single_c
    return mask_bedroom_only

def get_balcony_mask(
    blur={
        "Lab": np.zeros((1, 1, 3), dtype="uint8"),
        "HSV": np.zeros((1, 1, 3), dtype="uint8"),
    },
):
    balcony_boundary = {
        "bedroom_boundary": {
            "Lab": ([180, 120, 132], [254, 135, 165]),
            "HSV": ([10, 15, 200], [30, 110, 255]),
        }
    }
    bedroom_dark = {
        "bedroom_dark": {
            "Lab": ([160, 124, 139], [250, 130, 165]),
            "HSV": ([10, 30, 200], [30, 90, 250]),
        }
    }
    balcony_colors = {"balcony": {"Lab": ([240, 125, 130], [254, 135, 140])}}

    balcony_boundary_mask = get_color_mask(blur, balcony_boundary)
    bedroom_dark_mask = get_color_mask(blur, bedroom_dark)
    balcony_mask = get_color_mask(blur, balcony_colors)

    # remain balconies only

    mask_balcony_only = np.zeros_like(balcony_boundary_mask)

    contours, _ = cv2.findContours(
        balcony_boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for c in contours:
        mask_single_c = np.zeros_like(mask_balcony_only)
        cv2.drawContours(mask_single_c, [c], -1, 255, -1)

        c_area = cv2.countNonZero(mask_single_c)
        dark_area = cv2.countNonZero(mask_single_c & bedroom_dark_mask)
        balcony_area = cv2.countNonZero(mask_single_c & balcony_mask)

        if dark_area <= balcony_area and 10 <= balcony_area:
            mask_balcony_only |= mask_single_c
    return mask_balcony_only

def get_entrance_mask(bgr=np.zeros((1, 1, 3), dtype="uint8")):
    entrance_boundary = {"white_and_gray": {"HSV": ([0, 0, 170], [255, 20, 255])}}
    white = {"white": {"HSV": ([0, 0, 245], [255, 10, 255])}}
    gray = {"gray": {"HSV": ([0, 0, 230], [255, 10, 245])}}

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    blur = {}
    blur["Lab"] = cv2.bilateralFilter(lab, 15, 5, 150)
    blur["BGR"] = cv2.cvtColor(blur["Lab"], cv2.COLOR_Lab2BGR)
    blur["HSV"] = cv2.cvtColor(blur["BGR"], cv2.COLOR_BGR2HSV)

    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_5c = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    kernel_7c = np.zeros((7, 7), np.uint8)
    cv2.circle(kernel_7c, (3, 3), 3, 1, -1)
    kernel_9c = np.zeros((9, 9), np.uint8)
    cv2.circle(kernel_9c, (4, 4), 4, 1, -1)
    kernel_15c = np.zeros((15, 15), np.uint8)
    cv2.circle(kernel_15c, (7, 7), 7, 1, -1)

    mask_e, mask_w, mask_g = [
        color_dict_mask(blur, x) for x in [entrance_boundary, white, gray]
    ]
    area_e, area_w, area_g = [cv2.countNonZero(x) for x in [mask_e, mask_w, mask_g]]

    mask_e_e = cv2.erode(mask_e, kernel_7c)

    mask_w_d, mask_g_d = [cv2.dilate(x, kernel_15c) for x in [mask_w, mask_g]]
    mask_wg_c = cv2.erode(mask_w_d & mask_g_d, kernel_15c)

    #     if debug:
    #         print(area_e, area_w, area_g)
    #         plt.figure()
    #         plt.imshow(mask_e_e & 32 | mask_wg_c & 128, cmap="binary")

    contours, hierarchy = cv2.findContours(
        mask_e_e & mask_wg_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    mask_ent = np.zeros_like(mask_e)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cnt = contours[i]
            mask_c = np.zeros_like(mask_ent)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)

            area_c = cv2.countNonZero(mask_c & mask_e)
            area_c_w = cv2.countNonZero(mask_c & mask_w)
            area_c_g = cv2.countNonZero(mask_c & mask_g)

            if (
                area_c >= 100
                and area_c >= 0.01 * area_g
                and area_c_w >= 0.3 * area_c
                and area_c_g >= 0.3 * area_c
                and area_c_w + area_c_g >= 0.8 * area_c
            ):
                mask_ent |= mask_c

    mask_ent = cv2.morphologyEx(mask_ent, cv2.MORPH_CLOSE, kernel_15c)

    if debug:
        fig = plt.figure(figsize=(3, 3), dpi=300)
        plt.axes().axis("off")
        plt.imshow(mask_ent & 128, cmap="binary")
        plt.tight_layout()
    #         fig.savefig("floorplan_entrance.pdf", bbox_inches="tight", pad_inches=0)

    return mask_ent

def get_bathroom_mask(
    blur={
        "Lab": np.zeros((1, 1, 3), dtype="uint8"),
        "HSV": np.zeros((1, 1, 3), dtype="uint8"),
    },
):
    bathroom_colors = {"bathroom": {"HSV": ([90, 10, 220], [110, 40, 255])}}

    mask = get_color_mask(blur, bathroom_colors)

    return mask

def get_watershed(
    thresh=np.zeros((1, 1), dtype="uint8"), markers=np.zeros((1, 1), dtype="uint8")
):
    unknown = cv2.subtract(thresh, markers.astype(thresh.dtype))

    markers = markers.astype(np.int32)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(np.stack([thresh] * 3, axis=2), markers)
    markers = markers - 1
    markers[markers <= 0] = 0

    return markers

# In[4]:

# https://stackoverflow.com/questions/26537313/how-can-i-find-endpoints-of-binary-skeleton-image-in-opencv
def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1
    return out

# In[5]:

def get_unit_mask(bgr=np.zeros((1, 1, 3), dtype="uint8")):
    """Returns unit plan masks of the unit plan,
    as a dictionary of opencv masks and also a single combined mask,
    including masks for walls, entrances, LDK, bedrooms, balconies, and bathrooms."""

    AREA_UNIT = 128
    AREA_WALL = 64
    AREA_ENTRANCE = 32
    AREA_LDK = 16
    AREA_BEDROOM = 8
    AREA_BALCONY = 4
    AREA_BATHROOM = 2

    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_5c = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    kernel_7c = np.zeros((7, 7), np.uint8)
    cv2.circle(kernel_7c, (3, 3), 3, 1, -1)
    kernel_9c = np.zeros((9, 9), np.uint8)
    cv2.circle(kernel_9c, (4, 4), 4, 1, -1)
    kernel_15c = np.zeros((15, 15), np.uint8)
    cv2.circle(kernel_15c, (7, 7), 7, 1, -1)

    kernel_cross = np.array([[0, 1, 0,], [1, 1, 1,], [0, 1, 0,],], dtype=np.uint8,)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img = {"BGR": bgr, "RGB": rgb, "Lab": lab, "HSV": hsv}

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(rgb)
        plt.tight_layout()

    blur = {"Lab": cv2.bilateralFilter(lab, 15, 25, 150)}
    blur["BGR"] = cv2.cvtColor(blur["Lab"], cv2.COLOR_Lab2BGR)
    blur["RGB"] = cv2.cvtColor(blur["BGR"], cv2.COLOR_BGR2RGB)
    blur["HSV"] = cv2.cvtColor(blur["BGR"], cv2.COLOR_BGR2HSV)

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(blur["RGB"])
        plt.tight_layout()

    ######################################
    # Get wall/indoor/outdoor markers    #
    ######################################

    ### get wall

    wall_mask = get_wall_mask(bgr)
    wall_mask_d = cv2.dilate(wall_mask, kernel_9c)

    # entrance
    ent_mask = get_entrance_mask(bgr)
    ent_mask_d = cv2.dilate(ent_mask, kernel_9c)

    ### outside of the largest foreground area as outdoor boundary

    white_color = {"white": {"HSV": ([0, 0, 245], [180, 10, 255])}}
    white_mask = color_dict_mask({"HSV": blur["HSV"]}, white_color)

    ret, markers = cv2.connectedComponents(~white_mask)
    max_i = max(range(1, ret), key=lambda i: (markers == i).sum())
    #     print(max_i)
    mask = (markers == max_i).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_15c)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, 255, -1)

    outdoor_mask = cv2.morphologyEx(~mask, cv2.MORPH_CLOSE, kernel_9c)
    outdoor_mask_d = cv2.dilate(outdoor_mask, kernel_9c)

    #     if debug:
    #         fig = plt.figure(figsize=(6, 4), dpi=300)
    #         plt.axes().axis("off")
    #         plt.imshow(
    #             outdoor_mask, cmap="binary",
    #         )
    #         plt.tight_layout()

    #####################################
    # Getting color based masks         #
    #####################################

    #     wall_mask
    #     ent_mask

    ldk_mask = get_LDK_mask(blur)
    bed_mask = get_bedroom_mask(blur)
    bal_mask = get_balcony_mask(blur)
    bath_mask = get_bathroom_mask(blur)

    indoor_mask = ent_mask | ldk_mask | bed_mask | bal_mask | bath_mask

    ### get bounding box of indoor mask

    x, y, w, h = cv2.boundingRect(indoor_mask)
    indoor_bbox = cv2.rectangle(
        np.float32(np.zeros_like(indoor_mask)), (x, y), (x + w, y + h), 255, -1
    ).astype(np.uint8)

    ### make outmost zones do not contain LDK marker outdoor

    zones = ~outdoor_mask_d & ~wall_mask_d
    zones = cv2.dilate(zones, kernel_9c)

    ret, markers = cv2.connectedComponents(zones)
    for i in range(1, ret):
        marker = (markers == i).astype(np.uint8) * 255
        if not (marker & ldk_mask).sum() and (marker & outdoor_mask_d).sum():
            outdoor_mask |= marker

    ### regenerate masks

    outdoor_mask = cv2.morphologyEx(outdoor_mask, cv2.MORPH_CLOSE, kernel_9c)
    outdoor_mask_d = cv2.dilate(outdoor_mask, kernel_9c)

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            outdoor_mask, cmap="binary",
        )
        plt.tight_layout()

    #####################################
    # Skeleton of walls and space       #
    #####################################

    zones = ~wall_mask_d
    #     zones = cv2.dilate(zones, kernel_9c)

    skeleton, dist = medial_axis(zones, return_distance=True)
    skeleton = skeleton.astype(np.uint8) * 255
    ret, markers = cv2.connectedComponents(skeleton)

    skel_indoor = np.zeros_like(skeleton)
    for i in range(1, ret):
        marker = (markers == i).astype(np.uint8) * 255
        if cv2.countNonZero(marker & indoor_mask):
            skel_indoor |= marker

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            skel_indoor | (wall_mask & 32), cmap="binary",
        )
        plt.tight_layout()

    #####################################
    # Get non-wall borders              #
    #####################################

    border = cv2.Canny(blur["RGB"], 100, 200) & ~ent_mask_d

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            border, cmap="binary",
        )
        plt.tight_layout()

    ### pick borders touching walls and the skeleton

    ret, markers = cv2.connectedComponents(border)
    for i in range(1, ret):
        marker = (markers == i).astype(np.uint8) * 255
        if not ((marker & wall_mask).sum() and (marker & skel_indoor).sum()):
            border &= ~marker

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            border | wall_mask_d & 32, cmap="binary",
        )
        plt.tight_layout()

    ### if a white/gray space is larger than the smallest bedroom, it's outside

    #     # size of the smallest bedroom (for determine a core)
    #     min_bed_size = cv2.countNonZero(bed_mask)
    #     ret, markers = cv2.connectedComponents(
    #         cv2.morphologyEx(bed_mask, cv2.MORPH_CLOSE, kernel_9c) & ~wall_mask
    #     )
    #     for i in range(1, ret):
    #         marker = (markers == i).astype(np.uint8) * 255
    #         if cv2.countNonZero(marker) < min_bed_size:
    #             min_bed_size = cv2.countNonZero(marker)
    #     if debug:
    #         print(min_bed_size)

    zones = ~wall_mask & ~border
    zones = cv2.morphologyEx(zones, cv2.MORPH_OPEN, kernel_5c)
    ret, markers = cv2.connectedComponents(zones, connectivity=4)
    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(markers, cmap="gist_ncar")
        plt.tight_layout()

        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(markers % 20, cmap="tab20")
        plt.tight_layout()

    indoor_mask_area = cv2.countNonZero(indoor_mask)
    for i in range(1, ret):
        marker = (markers == i).astype(np.uint8) * 255
        if not (marker & indoor_mask).sum():
            if cv2.countNonZero(marker) > 0.10 * indoor_mask_area:
                outdoor_mask |= marker

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            outdoor_mask | wall_mask & 32, cmap="binary",
        )
        plt.tight_layout()

    ### add boundaries of color masks if a zone contains more than one color

    del outdoor_mask_d

    color_stacked = np.dstack(
        (outdoor_mask, ent_mask_d, ldk_mask, bed_mask, bal_mask, bath_mask,)
    )
    if debug:
        print(color_stacked.shape)
        print(
            (
                np.expand_dims(zones > 0, axis=2) & cv2.dilate(color_stacked, kernel_9c)
                > 0
            ).sum(axis=(0, 1))
        )

    edge_stacked = np.zeros_like(color_stacked)
    for k in range(6):
        edge_stacked[:, :, k] = cv2.Canny(color_stacked[:, :, k], 100, 200) & ~ent_mask
    edge_combined = np.bitwise_or.reduce(edge_stacked, 2)

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            edge_combined, cmap="binary",
        )
        plt.tight_layout()

    #     ret, markers = cv2.connectedComponents(zones, connectivity=4)
    for i in range(1, ret):
        marker = (markers == i).astype(np.uint8) * 255
        indoor_areas = (np.expand_dims(marker > 0, axis=2) & color_stacked).sum(
            axis=(0, 1)
        )
        if np.count_nonzero(indoor_areas) >= 2:
            border |= marker & edge_combined

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            border, cmap="binary",
        )
        plt.tight_layout()

    #####################################
    # Fill zones                        #
    #####################################

    wall_mask_3d = np.expand_dims(wall_mask, axis=2)
    wall_mask_d_3d = np.expand_dims(wall_mask_d, axis=2)

    color_stacked = (
        np.dstack((outdoor_mask, ent_mask_d, ldk_mask, bed_mask, bal_mask, bath_mask,))
        & ~wall_mask_3d
    )
    zones_filled = np.zeros_like(color_stacked)

    zones = ~wall_mask & ~border
    zones = cv2.morphologyEx(zones, cv2.MORPH_OPEN, kernel_5c)

    # remove area not touching indoor markers
    ret, markers = cv2.connectedComponents(~wall_mask)
    for i in range(1, ret):
        marker = (markers == i).astype(np.uint8) * 255
        if not ((marker & indoor_mask).sum()):
            zones &= ~marker

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            zones, cmap="binary",
        )
        plt.tight_layout()

    # make zones outside if more than a half of it is outside of bounding box (sanity check)

    ret, markers = cv2.connectedComponents(zones, connectivity=4)
    marker_stacked = np.dstack(
        [(markers == i).astype(np.uint8) * 255 for i in range(ret)]
    )
    indexes = list(range(1, ret))

    indoor_mask_area = cv2.countNonZero(indoor_mask)
    margin = 0.02 * indoor_mask_area

    for i in indexes:
        marker = marker_stacked[:, :, i]
        if cv2.countNonZero(marker) % 2 > (
            cv2.countNonZero(marker & indoor_bbox)  # + margin
        ):
            indexes.remove(i)
            zones &= ~marker

            # outdoor
            color_stacked[:, :, 0] |= marker
            zones_filled[:, :, 0] |= marker

    # fill
    count_last = len(indexes)
    remove_indexes = []
    repeat = 0
    while indexes:
        if debug:
            print(cv2.countNonZero(zones))

        for i in indexes:
            marker = marker_stacked[:, :, i]
            indoor_areas = (np.expand_dims(marker > 0, axis=2) & color_stacked > 0).sum(
                axis=(0, 1)
            )
            k = indoor_areas.argmax()

            if debug:
                print(i, k, indoor_areas[k])

            if indoor_areas[k]:
                if k != 0 or indoor_areas[1]:
                    remove_indexes.append(i)
                    zones &= ~marker

                    color_stacked[:, :, k] |= marker
                    zones_filled[:, :, k] |= marker

        indexes = [i for i in indexes if i not in remove_indexes]

        if len(indexes) == count_last:
            color_stacked = cv2.dilate(color_stacked, kernel_15c)
            color_stacked &= ~wall_mask_d_3d
            repeat += 1
        else:
            count_last = len(indexes)
            repeat = 0

            if debug:
                fig = plt.figure(figsize=(6, 4), dpi=300)
                plt.axes().axis("off")
                plt.imshow(
                    zones, cmap="binary",
                )
                plt.tight_layout()

                fig = plt.figure(figsize=(6, 4), dpi=300)
                plt.axes().axis("off")
                plt.imshow(
                    zones_filled[:, :, 0:3] | color_stacked[:, :, 0:3] & 128,
                    cmap="binary",
                )
                plt.tight_layout()

                fig = plt.figure(figsize=(6, 4), dpi=300)
                plt.axes().axis("off")
                plt.imshow(
                    zones_filled[:, :, 3:6] | color_stacked[:, :, 3:6] & 128,
                    cmap="binary",
                )
                plt.tight_layout()

        if repeat == 10:
            break

    if debug:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            zones_filled[:, :, 0:3], cmap="binary",
        )
        plt.tight_layout()

        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.axes().axis("off")
        plt.imshow(
            zones_filled[:, :, 3:6], cmap="binary",
        )
        plt.tight_layout()

    ### return wall instead of outdoor
    unit_comb = np.concatenate(
        (
            np.expand_dims(
                wall_mask
                & cv2.dilate(np.bitwise_or.reduce(zones_filled[:, :, 1:6], 2), kernel_15c),
                axis=2,
            ),
            zones_filled[:, :, 1:6],
        ),
        axis=-1,
    )

    ### return outdoor/entrance/LDK/bedroom/balcony/bathroom stacked mask
    return unit_comb

# # test and vis

# In[6]:

cv2.__version__

# In[15]:

bgr = read_bgr_from_image_unicode("/fp_img/15156_129.jpg")
# 9765_107A
# 1776_105
# 102487_266B
# 2672_162
# 16429_107

# In[16]:

unit_comb = get_unit_mask(bgr)

# In[17]:

np.amax(unit_comb), np.amin(unit_comb), unit_comb.shape, unit_comb.dtype

# In[18]:

for i in range(6):
    plt.figure()
    plt.imshow(unit_comb[:,:,i])
```

11_floorplan_normalization.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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

debug = True  # plot every steps

# # import floorplan analysis

# In[56]:

from floorplan_analysis import read_bgr_from_image_unicode, get_unit_mask
from floorplan_analysis import rescale_fp
from floorplan_analysis import mono_fp, fp_float_from_mono, fp_uint8_from_mono
from floorplan_analysis import pad_fp

# # normalization

# In[63]:

bgr = read_bgr_from_image_unicode("/fp_img/107903_113G.jpg")

# 101160_113E 85
# 103915_112C 85
# 104127_107B 80
# 107903_113G 85
# 108838_117B 85

plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# In[64]:

unit_comb_orig = get_unit_mask(bgr)

# In[98]:

fp = rescale_fp(unit_comb_orig, 85)
mono = mono_fp(unit_comb)
fp = fp_uint8_from_mono(mono)

# In[99]:

fp.shape

# In[100]:

fp.sum(axis=(0, 1))

# In[101]:

for i in range(6):
    plt.figure()
    plt.imshow(fp[:, :, i])

# In[105]:

plt.imshow(fp[:, :, [0, 1, 4]])

# In[104]:

ret, markers = cv2.connectedComponents(fp[:, :, 4])
plt.imshow(markers)

# In[39]:

dist_transform = cv2.distanceTransform(unit_comb[:, :, 3], cv2.DIST_L2, 5)
plt.imshow(dist_transform)

# In[40]:

dist_transform.max()

# In[41]:

plt.imshow(dist_transform == dist_transform.max())

# In[27]:

markers[dist_transform == dist_transform.max()]

# In[ ]:

# In[36]:

bal_moments = cv2.moments(unit_comb[:, :, 4], True)
ind_moments = cv2.moments(np.bitwise_or.reduce(unit_comb, 2), True)

### make it down

if bal_moments["m00"] > 0:
    if (
        bal_moments["nu02"] / ind_moments["nu02"]
        > bal_moments["nu20"] / ind_moments["nu20"]
    ):
        #         print("down")
        pass
    elif (
        bal_moments["m10"] / bal_moments["m00"]
        < ind_moments["m10"] / ind_moments["m00"]
    ):
        #         print("left")
        unit_comb = np.rot90(unit_comb, 1)
    else:
        #         print("right")
        unit_comb = np.rot90(unit_comb, -1)
else:
    ldk_moments = cv2.moments(~unit_comb[:, :, 3], True)
    if ldk_moments["nu02"] >= ldk_moments["nu20"]:
        # print("down")
        pass
    elif (
        ldk_moments["m10"] / ldk_moments["m00"]
        < ind_moments["m10"] / ind_moments["m00"]
    ):
        # print("left")
        unit_comb = np.rot90(unit_comb, 1)
    else:
        # print("right")
        unit_comb = np.rot90(unit_comb, -1)

# In[74]:

plt.imshow(unit_comb[:, :, 1])

# In[71]:

### put entrance to left

ent_moments = cv2.moments(unit_comb[:, :, 1], True)
if ent_moments["m00"] and (
    (ent_moments["m10"] / ent_moments["m00"]) > (unit_comb.shape[1] / 2)
):
    # print("flip")
    unit_comb = np.flip(unit_comb, axis=1)

# In[72]:

ent_moments["m10"] / ent_moments["m00"], ent_moments["m01"] / ent_moments["m00"]

# In[75]:

plt.imshow(unit_comb[:, :, 1])

# In[36]:

area = 85.0
target_ppm = 5  # pixels per meter

# indoor pixels excluding balcony
pixels = cv2.countNonZero(np.bitwise_or.reduce(unit_comb, 2) & ~unit_comb[:, :, 4])

print(area, pixels)

scale = sqrt(area * target_ppm ** 2 / pixels)
print(scale)

unit_scale = rescale(unit_comb, scale, mode="edge", multichannel=True)
plt.imshow(unit_scale[:, :, 0])

# In[37]:

indexes = np.where(unit_scale != 0)

# In[17]:

unit_clipped = unit_scale[
    min(indexes[0]) : max(indexes[0]) + 1, min(indexes[1]) : max(indexes[1]) + 1
]
plt.imshow(unit_clipped[:, :, 4])

# In[18]:

def align_fp(unit_comb):
    """put the main side to down and entrance to left"""

    def align_calc(unit_comb):
        def align_calc_x(unit_comb, cx):
            moment_half = cv2.moments(
                np.bitwise_or.reduce(unit_comb[:, cx:, :], 2), True
            )
            moment_bal = cv2.moments(unit_comb[:, cx:, 4], True)
            return moment_bal["m20"] / moment_half["m20"] * moment_half["m00"]

        def align_calc_y(unit_comb, cy):
            moment_half = cv2.moments(
                np.bitwise_or.reduce(unit_comb[cy:, :, :], 2), True
            )
            moment_bal = cv2.moments(unit_comb[cy:, :, 4], True)
            return moment_bal["m02"] / moment_half["m02"] * moment_half["m00"]

        x, y, w, h = cv2.boundingRect(np.bitwise_or.reduce(unit_comb, 2))
        cx, cy = x + w // 2, y + h // 2

        right, bottom = align_calc_x(unit_comb, cx), align_calc_y(unit_comb, cy)

        x, y, w, h = cv2.boundingRect(
            np.bitwise_or.reduce(cv2.rotate(unit_comb, cv2.ROTATE_180), 2)
        )
        cx, cy = x + w // 2, y + h // 2

        left, top = (
            align_calc_x(cv2.rotate(unit_comb, cv2.ROTATE_180), cx),
            align_calc_y(cv2.rotate(unit_comb, cv2.ROTATE_180), cy),
        )

        return top, bottom, left, right

    ind_moments = cv2.moments(np.bitwise_or.reduce(unit_comb, 2), True)
    ent_moments = cv2.moments(unit_comb[:, :, 1], True)

    if cv2.countNonZero(unit_comb[:, :, 4]):
        result = np.array(align_calc(unit_comb))
        # print(result)
        side = np.argmax(result)

        if side == 2:
            # print("left")
            unit_comb = np.rot90(unit_comb, 1)
        elif side == 3:
            # print("right")
            unit_comb = np.rot90(unit_comb, -1)
        else:
            # print("bottom?")
            pass

    else:
        ldk_moments = cv2.moments(~unit_comb[:, :, 3], True)

        if ldk_moments["nu02"] >= ldk_moments["nu20"]:
            # print("down")
            pass
        elif (
            ldk_moments["m10"] / ldk_moments["m00"]
            < ent_moments["m10"] / ent_moments["m00"]
        ):
            # print("left")
            unit_comb = np.rot90(unit_comb, 1)
        else:
            # print("right")
            unit_comb = np.rot90(unit_comb, -1)

    ### put entrance to left
    if (
        ent_moments["m10"] / ent_moments["m00"]
        > ind_moments["m10"] / ind_moments["m00"]
    ):
        # print("flip")
        unit_comb = np.flip(unit_comb, axis=1)

    return unit_comb

# In[19]:

from skimage.transform import rescale

def rescale_fp(unit_comb, area, target_ppm=5, trim=True):
    # indoor pixels excluding balcony
    pixels = cv2.countNonZero(np.bitwise_or.reduce(unit_comb, 2) & ~unit_comb[:, :, 4])
    # print(area, pixels)

    scale = sqrt(area * target_ppm ** 2 / pixels)
    # print(scale)

    unit_scale = rescale(unit_comb, scale, mode="edge", multichannel=True)

    if trim:
        indexes = np.where(unit_scale != 0)
        unit_scale = unit_scale[
            min(indexes[0]) : max(indexes[0]) + 1, min(indexes[1]) : max(indexes[1]) + 1
        ]

    return (unit_scale * 255).astype(np.uint8)

# In[20]:

rescale_fp(unit_comb, 85)
```

12_floorplan_image.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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

from math import sqrt, log10

import matplotlib.pyplot as plt

from os.path import expanduser, splitext
from os import scandir, makedirs

# import random

import csv

from tqdm import tnrange, tqdm_notebook

from pathlib import Path

debug = False  # plot every steps

# In[2]:

from floorplan_analysis import read_bgr_from_image_unicode, get_unit_mask
from floorplan_analysis import align_fp, rescale_fp

# In[3]:

cv2.__version__

# In[4]:

bgr = read_bgr_from_image_unicode("/fp_img/10001_57B.jpg")
# 9765_107A 누워있는
# 1776_105 코어
# 102487_266B 비사각

# 199_86
# 6_87
# 2_63
# 8_99

# 107323_110B 흰 영역 날아감

# 10001_57B 발코니 없음

# In[5]:

plt.imshow(bgr)

# In[6]:

unit_comb_orig = get_unit_mask(bgr)

# In[7]:

plt.imshow(np.bitwise_or.reduce(unit_comb_orig, 2))

# In[8]:

cv2.countNonZero(unit_comb_orig[:, :, 4])

# In[9]:

unit_comb = align_fp(unit_comb_orig.copy()[:, :, :])
unit_comb = rescale_fp(unit_comb, 85)

plt.imshow(unit_comb[:, :, [0, 1, 4]])

# In[ ]:

# In[ ]:

# In[10]:

# AREA_WALL = 64
# AREA_ENTRANCE = 32
# AREA_LDK = 16
# AREA_BEDROOM = 8
# AREA_BALCONY = 4
# AREA_BATHROOM = 2

mask_bits = np.array([64, 32, 16, 8, 4, 2], dtype=np.uint8)

# In[18]:

# binary cut value
cut = 0.01 * np.ones(6)
for i in range(6):
    if cv2.countNonZero(unit_comb[:, :, i]):
        cut[i] = (
            np.average(unit_comb[:, :, i][np.nonzero(unit_comb[:, :, i] > 0)]) - 0.01
        )

# In[19]:

plt.imshow((unit_comb > cut)[:, :, 4])

# In[20]:

mono = ((unit_comb > cut) * 255) & mask_bits

# In[21]:

mono.shape, mono.dtype

# In[22]:

np.amax(mono[:, :, 2])

# In[23]:

mono = np.bitwise_or.reduce(mono, 2)
plt.imshow(mono)

# In[24]:

def mono_fp(unit_comb):
    """create bit mask image from
    wall/entrance/LDK/bedroom/balcony/bathroom stacked array"""

    # AREA_WALL = 64
    # AREA_ENTRANCE = 32
    # AREA_LDK = 16
    # AREA_BEDROOM = 8
    # AREA_BALCONY = 4
    # AREA_BATHROOM = 2

    mask_bits = np.array([64, 32, 16, 8, 4, 2], dtype=np.uint8)

    # binary cut value
    cut = 0.01 * np.ones(6)
    for i in range(6):
        if cv2.countNonZero(unit_comb[:, :, i]):
            cut[i] = (
                np.average(unit_comb[:, :, i][np.nonzero(unit_comb[:, :, i] > 0)])
                - 0.01
            )

    mono = ((unit_comb > cut) * 255).astype(np.uint8) & mask_bits
    mono = np.bitwise_or.reduce(mono, 2)
    return mono
```

15_process_floorplan_images.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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

# In[21]:

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
files_IDs_exclude.append(Path('processed.csv'))

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

# In[22]:

from multiprocessing import Pool

makedirs(dir_to, exist_ok=True)

with Pool(7) as p:
    p.starmap(process_floorplan_mono, zip(paths_from, area_list, IDs))

#     ls /data/fp_img_processed/ | wc -l
```

15_repeat_postprocess.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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

from floorplan_analysis import read_from_csv
from floorplan_analysis import read_bgr_from_image_unicode, get_unit_mask
from floorplan_analysis import align_fp, rescale_fp
from floorplan_analysis import mono_fp
from floorplan_analysis import read_mono_from_image_unicode, save_mono_to_image_unicode
from floorplan_analysis import fp_float_from_mono, fp_uint8_from_mono

# # postprocess files

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
        pass

# In[4]:

def realign_on_mono(
    path_from, area, filename_to, dir_to="/data/fp_img_processed/", ext_to=".png"
):
    try:
        mono = read_mono_from_image_unicode(path_from)
        unit_comb = fp_uint8_from_mono(mono)

        unit_comb = align_fp(unit_comb)

        mono = mono_fp(unit_comb)
        save_mono_to_image_unicode(mono, dir_to + filename_to + ext_to, ext_to)
    except:
        pass

# In[5]:

bgr = read_bgr_from_image_unicode("/fp_img/15079_105.jpg")
plt.imshow(bgr)

# In[6]:

realign_on_mono("/data/fp_img_processed/15079_105.png", 85, "mono", dir_to="")

# In[7]:

plt.imshow(read_mono_from_image_unicode("mono.png"))

# In[8]:

mono = read_mono_from_image_unicode("mono.png")
plt.imshow(mono)

# In[9]:

unit_comb = fp_uint8_from_mono(mono)
plt.imshow(unit_comb[:, :, [1, 3, 5]])

# In[10]:

unit_comb = align_fp(unit_comb)
plt.imshow(unit_comb[:, :, [1, 3, 4]])

# In[11]:

ret, markers = cv2.connectedComponents(unit_comb[:, :, 3])
plt.imshow(markers)

# In[12]:

markers

# In[13]:

dist_transform = cv2.distanceTransform(unit_comb[:, :, 3], cv2.DIST_L2, 5)
dist_transform.max()

# In[14]:

plt.imshow(dist_transform)

# In[15]:

plt.imshow(dist_transform == dist_transform.max())

# In[16]:

markers[dist_transform == dist_transform.max()]

# In[17]:

counts = np.bincount(markers[dist_transform == dist_transform.max()])
print(np.argmax(counts))

# In[18]:

mbr = (markers == np.argmax(counts))
plt.imshow(mbr)

# In[19]:

mbr.sum(axis=1).shape

# In[20]:

np.flip(mbr).sum(axis=1).argmax()

# In[21]:

mbr.sum(axis=1).argmax()

# In[22]:

np.flip(unit_comb[:, :, 5]).sum(axis=1).argmax()

# In[23]:

np.flip(unit_comb[:, :, 5]).sum(axis=1).argmax() - np.flip(mbr).sum(axis=1).argmax()

# In[24]:

(unit_comb[:, :, 5]).sum(axis=1).argmax() - (mbr).sum(axis=1).argmax()

# # main

# In[25]:

dir_ID_from = "/data/fp_img_processed/"
dir_IDs_exclude = "/data/exclude/"

dir_from = "/data/fp_img_processed/"

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

# In[26]:

list(ID_ext_dict.items())[:10]

# In[27]:

files_IDs_exclude = list(Path(expanduser(dir_IDs_exclude)).glob("*.csv"))
print(files_IDs_exclude)

# In[28]:

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

# In[29]:

import pandas as pd

path_csv = "fp_refined.csv"

df = pd.read_csv(path_csv)
df = df.set_index("id_after")
df

# In[30]:

ID_set = set(ID_ext_dict.keys()).difference(IDs_excl)
ID_set = ID_set.intersection(df.index)
IDs = list(ID_set)
print(len(IDs), "floorplans to go")

# In[31]:

paths_from = [dir_from + ID + ID_ext_dict[ID] for ID in IDs]
print(paths_from[:10])

# In[32]:

df.Area

# In[33]:

area_list = [df.Area[ID] for ID in IDs]
area_list[:10]

# # re-do the align

# In[34]:

from multiprocessing import Pool

makedirs(dir_to, exist_ok=True)

with Pool(7) as p:
    p.starmap(realign_on_mono, zip(paths_from, area_list, IDs))

#     ls /data/fp_img_processed/ | wc -l

# In[35]:

get_ipython().system('ls /data/fp_img_processed/8516_87* -al')
```

16_processed_list.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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
ID_path_dict = {
    splitext(f.name)[0]: f.path
    for f in scandir(dir_from)
    if f.is_file()
}
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
```

17_image_size.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('matplotlib', 'inline')

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

csv_to = "size.csv"

### all of the plans
ID_path_dict = {splitext(f.name)[0]: f.path for f in scandir(dir_from) if f.is_file()}
print(len(ID_path_dict.keys()), "floorplans")

# In[4]:

list(ID_path_dict.items())[:10]

# In[5]:

with open(csv_to, "w", newline="", encoding="utf-8-sig") as csvfile:
    listwriter = csv.writer(csvfile)

    IDs_error = []
    for ID, path in tqdm(ID_path_dict.items(), desc="Processing plans"):
        try:
            with Image.open(path) as im:
                width, height = im.size
            listwriter.writerow([ID, width, height])
        except:
            IDs_error.append(ID)
    print(len(IDs_error))
    print(IDs_error)

# # analysis

# In[6]:

import pandas as pd

df = pd.read_csv(csv_to, names=["ID", "width", "height"])
df = df.set_index("ID")

# In[7]:

df

# In[8]:

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(11, 5), dpi=300)

df.width.plot.kde(ax=axs[0])
axs[0].set_title("Width of unit plans (m)")

df.height.plot.kde(ax=axs[1])
axs[1].set_title("Depth of unit plans (m)")

for ax in axs:
    ax.set_xlim(0, 120)
    ax.set_xticks(range(0, 120+1, 20))
    ax.set_xticklabels(range(0, 24+1, 4))
    ax.set_ylim(0)
    ax.set_yticks([])

plt.tight_layout()
fig.savefig("fp_size_kde.pdf", bbox_inches="tight", pad_inches=0)
```

20_floorplan_dataset_tfrecord.py

```python
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

from tqdm.notebook import tqdm, tnrange

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

from floorplan_analysis import read_mono_from_image_unicode
from floorplan_analysis import fp_float_from_mono

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

# 10001_57B
# 2112_49_0
mono = read_mono_from_image_unicode("/data/fp_img_processed/2112_49_0.png")
fp = fp_float_from_mono(mono)
fp = pad_fp(fp)
plt.imshow(fp[..., [0, 2, 4]])

# In[8]:

def preprocess_fp(path):
    mono = read_mono_from_image_unicode(path)
    fp = fp_float_from_mono(mono)
    fp = pad_fp(fp)
    return fp

# In[9]:

fp.shape, fp.dtype, np.amax(fp)

# In[10]:

plt.imshow(visualize_fp(fp))

# In[11]:

def preprocess_year(year):
    return max(0, (year - 1970) // 5)

# In[12]:

print([preprocess_year(year) for year in range(1969, 2022)])

# In[13]:

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
df_tfrecord

# In[14]:

rows = df_tfrecord.sample(frac=1).values

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

# In[15]:

# tf.train.Feature(float_list=tf.train.FloatList(value=processed.flatten()))

# _floats_array_feature(fp)

# In[16]:

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

# In[17]:

example_proto = tf.train.Example.FromString(serialize_example(rows[0]))
print(repr(example_proto)[:100])
print(repr(example_proto)[-700:])

# In[18]:

path_all_tfrecord = "fp.tfrecord"

options_gzip = tf.io.TFRecordOptions(compression_type="GZIP")

errors = []

for (path, rows) in zip([path_all_tfrecord], [rows]):
    with tf.io.TFRecordWriter(path=path, options=options_gzip) as writer:
        for row in tqdm(rows, desc="Processing plans"):
            try:
                serialized_example = serialize_example(row)
                writer.write(serialized_example)
            except:
                errors.append(row[1])

if errors:
    print(errors)

# In[19]:

raw_dataset = tf.data.TFRecordDataset(path_all_tfrecord, compression_type="GZIP")
raw_dataset

# In[20]:

for raw_record in raw_dataset.take(2):  # WARNING: deprecated
    print(repr(raw_record)[:400])

# In[21]:

fp_dim = (112, 112, 6)

def _parse_function(example_proto):
    # Create a description of the features.
    feature_description = {
        "floorplan": tf.io.FixedLenFeature(
            fp_dim, tf.float32, default_value=tf.zeros(fp_dim, tf.float32)
        ),
        "plan_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "year": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "sido": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "norm_area": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "num_rooms": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "num_baths": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    return parsed_example

parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset

# In[22]:

for parsed_record in parsed_dataset.take(3):
    print(repr(parsed_record))
```

24_dataset_all_train_test_56.py

```python
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
```

41_visualize_floorplan.py

```python
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

        fp_light = cv2.cvtColor(fp_rgba.astype(np.uint8)*255, cv2.COLOR_RGB2Lab)[:, :, 0] / 100
        fp_rgba = pad_fp(fp_rgba, fp_light.shape[1], fp_light.shape[0])

        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(fp_rgba)
        plt.axis("off")

        plt.tight_layout()
        fig.savefig(f"{dir_to}/{id}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

visualize_from_file(['10001_57B'])

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
```

61_VGG_CAM.py

```python
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
```

62_VGG_CAM_prediction.py

```python
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
    zip(ids, year_true, predictions), columns=["ID", "true", "prediction"],
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

fig.savefig(
    "vgg_5y_heatmap.pdf", bbox_inches="tight", pad_inches=0,
)
fig.savefig(
    "vgg_5y_heatmap.png", bbox_inches="tight", pad_inches=0,
)

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
```

63_VGG_CAM_activation.py

```python
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
```

65_biclustering.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# In[2]:

df = pd.read_csv("vgg_5y_activation.csv.gz", index_col=0)
df

# 신경망 출력값의 범위

# In[3]:

flatten = pd.DataFrame(df.to_numpy().reshape(-1))
flatten.sample(frac=0.01, random_state=1106).plot.kde()

# In[4]:

flatten.describe()

# 바이클러스터링 군집 수 결정

# In[5]:

from sklearn.cluster import MiniBatchKMeans

# In[6]:

clust = MiniBatchKMeans(
    n_clusters=30, batch_size=100, verbose=0, random_state=1106,
).fit(df)

clust.inertia_

# In[7]:

inertia_dict = {
    i: MiniBatchKMeans(n_clusters=i, random_state=1106,).fit(df).inertia_
    for i in range(2, 31)
}

# In[8]:

inertia_dict

# In[293]:

fig = plt.figure(figsize=(10, 3), dpi=300)

ax = pd.Series(inertia_dict).plot()
ax.set_xlim(0, 30)
ax.set_ylim(0)

ax2 = ax.twinx()
(pd.Series(inertia_dict).diff() * (-1)).plot(c="tab:red", ax=ax2)
# ax2.set_ylim(0)
ax2.axhline(0, c="tab:red", ls=":")
ax2.tick_params(axis="y", labelcolor="tab:red")

n_clusters = 16  # <-------------------------

[ax.axvline(n + 0.5, c="k", ls="--") for n in [n_clusters]]
ax.set_xticks(list(range(0, 30 + 1, 10)) + [n_clusters])

plt.show()
fig.savefig("biclust_n.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_n.pdf", bbox_inches="tight", pad_inches=0)

#

# In[12]:

import matplotlib.colors as colors

fig = plt.figure(figsize=(5, 7), dpi=300)
ax = fig.gca()

im = ax.matshow(df[::100], vmin=0, vmax=1, cmap="viridis",)

ax.set_yticks(range(0, 500 + 1, 100))
ax.set_yticklabels([f"{t:,}" for t in range(0, 50000 + 1, 10000)])

fig.colorbar(im, shrink=0.5)

plt.show()
fig.savefig("biclust_before.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_before.pdf", bbox_inches="tight", pad_inches=0)

# In[13]:

df_refined = pd.read_csv("fp_refined.csv")
df_refined

# In[14]:

df_vgg = pd.read_csv("vgg_5y_prediction.csv")

# In[294]:

df_clust = df_vgg.join(
    df_refined.set_index("id_after")[
        ["year", "sido_cluster_code", "Area", "Rooms", "Baths"]
    ],
    on="ID",
)
df_clust

# In[295]:

n_clusters = 16

# In[296]:

from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=n_clusters, random_state=1106, n_jobs=6).fit(df)

df_clust["cluster"] = model.row_labels_
order = df_clust.groupby("cluster")["year"].mean().argsort().values
rank = order.argsort()
df_clust["cluster"] = rank[df_clust.cluster]

df_clust.groupby("cluster").year.mean()

# In[297]:

fig = plt.figure(figsize=(10, 3), dpi=300)

axs = df_clust.groupby("cluster")["year"].plot.kde(bw_method=0.5)
axs[0].set_xlim(1969, 2020)
axs[0].set_ylim(0)

# plt.legend()

# In[300]:

ordered = df.iloc[
    np.argsort(rank[model.row_labels_]), np.argsort(rank[model.column_labels_])
]

fig = plt.figure(figsize=(5, 7), dpi=300)
ax = fig.gca()

im = ax.matshow(ordered[::100], vmin=0, vmax=1, cmap="viridis")

ax.set_yticks(range(0, 500 + 1, 100))
ax.set_yticklabels([f"{t:,}" for t in range(0, 50000 + 1, 10000)])

fig.colorbar(im, shrink=0.5)

plt.show()
fig.savefig("biclust_after.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_after.pdf", bbox_inches="tight", pad_inches=0)

# In[301]:

df_mean = (
    df.groupby(df_clust["cluster"])
    .mean()
    .groupby(rank[model.column_labels_], axis=1)
    .mean()
)

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()

im = ax.matshow(df_mean)
fig.colorbar(im)

ax.set_xticks(range(n_clusters))
ax.set_yticks(range(n_clusters))

ax.set_xticklabels(range(1, n_clusters + 1))
ax.set_yticklabels(range(1, n_clusters + 1))

plt.show()
fig.savefig("biclust_mean.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_mean.pdf", bbox_inches="tight", pad_inches=0)

# In[465]:

from matplotlib.colors import DivergingNorm

df_corr = df.groupby(rank[model.column_labels_], axis=1).sum().corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_corr, dtype=np.bool))

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()
im = ax.matshow(
    np.ma.array(df_corr, mask=mask), cmap="coolwarm", norm=DivergingNorm(0),
)

fig.colorbar(im)

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.xaxis.tick_bottom()

ax.set_xticks(range(n_clusters))
ax.set_yticks(range(n_clusters))

ax.set_xticklabels(range(1, n_clusters + 1))
ax.set_yticklabels(range(1, n_clusters + 1))

plt.show()
fig.savefig("biclust_corr.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_corr.pdf", bbox_inches="tight", pad_inches=0)

# In[488]:

plt.matshow(np.ma.array((df_corr >= 0.5), mask=mask))

ax = plt.gca()
ax.set_xticks(range(n_clusters))
ax.set_yticks(range(n_clusters))

ax.set_xticklabels(range(1, n_clusters + 1))
ax.set_yticklabels(range(1, n_clusters + 1))

plt.show()

# 1번부터 16번까지의 평면 계획 특성 요인은 각 요인과 짝지어져 군집화된 평면들의 평균 준공연도를 기준으로 정렬한 것이다.
# 인접한? 요인들은 서로 비슷한 시기의 평면 계획 특성을 나타낸다.
# 각 시기에 해당하는 평면에서는 해당 시기의 여러 계획 특성이 함께 나타나기 때문에,
# 서로 인접한 평면 계획 특성 요인들끼리는 양의 상관관계를 보인다.
#
# 5와 12, 6과 15는 서로 다른 시기의 평면에서 나타나는 계획 특성 요인이지만 양의 상관관계를 보인다.
# 대부분의 요인이 서로 상관관계가 없거나 음의 상관관계를 보이는 것과 다르다.
# 해당 평면 군집 사이에 계획적 공통점이 있다는 점을 시사한다.

# In[498]:

plt.matshow(np.ma.array(df_corr <= -0.3, mask=mask),)

ax = plt.gca()
ax.set_xticks(range(n_clusters))
ax.set_yticks(range(n_clusters))

ax.set_xticklabels(range(1, n_clusters + 1))
ax.set_yticklabels(range(1, n_clusters + 1))

plt.show()

# 1990년대 평면을 포괄하는 1--7번 요인과
# 2010년대 평면을 포괄하는 13--16번 요인 사이에서는
# 대다수에서 역의 상관관계가 두드러진다.
# 딥 러닝 모형이 학습한 1990년대와 2010년대의 평면 계획 특성은
# 서로 다른 시기의 계획 특성을 잘 구분할 수 있다.
#
# 한편,
# 3, 6, 15번 요인은
# 2000년대 평면을 포함하는 8--12번 요인과 역의 상관관계를 보인다.
# 2000년대 평면에서 잘 나타나지 않았던 계획 경향을 나타낸다.

# In[305]:

rank[model.column_labels_]

# In[306]:

np.savetxt("biclust_col.txt", rank[model.column_labels_], "%.u")

# In[307]:

df_clust.to_csv("biclust.csv")

# In[308]:

pd.crosstab(df_clust.cluster, df_clust.Rooms)

# In[309]:

pd.crosstab(df_clust.cluster, df_clust.Rooms, normalize="index")

# In[360]:

from matplotlib.ticker import PercentFormatter

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()

pd.crosstab(df_clust.cluster, df_clust.Rooms, normalize="index").plot.bar(
    stacked=True, ax=ax
).legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

ax.set_xticks(range(n_clusters))
ax.set_xticklabels(range(1, n_clusters + 1))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.savefig("biclust_rooms.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_rooms.pdf", bbox_inches="tight", pad_inches=0)

# 대부분에서 침실 3개가 가장 많고 그 다음으로 침실 4개가 2위인 것이 일반적.
#
# 13번은 침실 1개가 가장 많음.
#
# 0, 2번은 침실 3개 다음으로 침실 2개가 뒤따름. (2위)

# In[361]:

from matplotlib.ticker import PercentFormatter

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()

pd.crosstab(df_clust.cluster, df_clust.Baths, normalize="index").plot.bar(
    stacked=True, ax=ax
).legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

ax.set_xticks(range(n_clusters))
ax.set_xticklabels(range(1, n_clusters + 1))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.savefig("biclust_baths.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_baths.pdf", bbox_inches="tight", pad_inches=0)

# 대체로 화장실 2개가 가장 많지만,
# 침실 1개가 가장 많은 13번과
# 침실 2개가 2위인 0, 2번은
# 화장실 1개가 가장 많음.

# In[312]:

cmap = plt.get_cmap("tab20")
colors = cmap(np.linspace(0, 0.5 - 1 / cmap.N, cmap.N // 2))
colors

# In[313]:

from matplotlib.colors import LinearSegmentedColormap

tab20half = LinearSegmentedColormap.from_list("tab20 Lower Half", colors)

# In[362]:

from matplotlib.ticker import PercentFormatter

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()

pd.crosstab(df_clust.cluster, df_clust.true, normalize="index").plot.bar(
    stacked=True, cmap=tab20half, ax=ax
).legend(
    [
        "1969-74",
        "1975-79",
        "1980-84",
        "1985-89",
        "1990-94",
        "1995-99",
        "2000-04",
        "2005-09",
        "2010-14",
        "2015-19",
    ],
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
)

ax.set_xticks(range(n_clusters))
ax.set_xticklabels(range(1, n_clusters + 1))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.savefig("biclust_year.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_year.pdf", bbox_inches="tight", pad_inches=0)

# 각 군집이 서로 다른 시기에 따라 잘 묶임

# In[315]:

tab20_sido = LinearSegmentedColormap.from_list(
    "9 colors from tab20",
    plt.get_cmap("tab20")([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.50]),
)

# In[363]:

from matplotlib.ticker import PercentFormatter

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()

pd.crosstab(df_clust.cluster, df_clust.sido_cluster_code, normalize="index").plot.bar(
    stacked=True, ax=ax, cmap=tab20_sido,
).legend(
    [
        "Seoul",
        "Gyeonggi",
        "Incheon",
        "Gangwon",
        "Daejeon/Sejong/Chungcheong",
        "Busan/Ulsan/Gyeongsangnam",
        "Daegu/Gyeongsangbuk",
        "Gwangju/Jeolla",
        "Jeju",
    ],
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
)

ax.set_xticks(range(n_clusters))
ax.set_xticklabels(range(1, n_clusters + 1))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.savefig("biclust_sido.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_sido.pdf", bbox_inches="tight", pad_inches=0)

# In[317]:

df_clust.Area

# In[318]:

pd.cut(df_clust.Area, [0, 50, 60, 85, np.inf])

# In[319]:

pd.crosstab(
    df_clust.cluster, pd.cut(df_clust.Area, [0, 50, 60, 85, np.inf]), normalize="index"
)

# In[364]:

from matplotlib.ticker import PercentFormatter

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.gca()

pd.crosstab(
    df_clust.cluster, pd.cut(df_clust.Area, [0, 50, 60, 85, np.inf]), normalize="index"
).plot.bar(stacked=True, ax=ax).legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

ax.set_xticks(range(n_clusters))
ax.set_xticklabels(range(1, n_clusters + 1))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.savefig("biclust_area.png", bbox_inches="tight", pad_inches=0)
fig.savefig("biclust_area.pdf", bbox_inches="tight", pad_inches=0)
```

67_VGG_CAM_visualization.py

```python
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

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS,)
# heatmap[cam < 0.2] = 0
plot_bgr(heatmap)

# In[22]:

cam = get_biclust_cam(fp, 3)

cam = cv2.resize(cam, (56, 56))
print(cam.max())
cam /= cam.max()
cam[cam <= 0] = 0

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS,)
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

coords_row = np.insert(np.cumsum(heights), 0, 0,)[:-1]
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

# In[48]:

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

df_sample[(df_sample.cluster == 13)]

# 101160_113E
# 103915_112C
# 104127_107B
# 107903_113G
# 108838_117B

# In[159]:

def plot_bgr_scale(img):
    size_x, size_y = img.shape[:2]
    fig = plt.figure(figsize=(2 * size_x / 112, 2 * size_y / 112), dpi=300)
    plt.axes().axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()

ir, ic, i = 15,2,10
u_single = 84  # 56 84 112
df_clust = df_sample[(df_sample.cluster == ir) & (df_sample.area_group == ic)]
r = i // widths[ic]
c = i - r * widths[ic]
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
```

\normalsize
