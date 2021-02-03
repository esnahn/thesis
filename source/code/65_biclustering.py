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


# # 면적
# 
# 대다수는 국민주택 규모가 가장 많다.
# 3, 6, 7, 15에서 소형 이하가 (< 60) 과반을 차지한다.
# 10, 11, 13은 대형 (85 초과) 평면이 과반이다.
# 
# 몇몇 예외 말고는 면적 규모에 따른 분류가 이루어지지 않았다.
# 다양한 규모의 단위평면에서 나타나는 거시적인 변화를 나타낸다.

# # 총평
# 
# 시기별로 잘 분류함
# 
# 1990년대까지의 아파트 평면은 1--7번, 2000년대는 4--14번, 2010년대는 12--16번으로 중첩되어 나타남.
# 하나의 평면 유형이 여러 시기에 걸쳐 나타났다가 사라지는 모습을 보여줌.
# 
# 대체로 시기 구분과 관련없는 지역이나 면적 규모에 따른 차이는 잘 학습되지 않았음.
# 
# 시기적으로 의미 있는 규모 변화는 잘 파악이 되었음.
# 15번은 원룸형 도시형생활주택 (2009년 처음 등장)을 잘 찾아냄.
