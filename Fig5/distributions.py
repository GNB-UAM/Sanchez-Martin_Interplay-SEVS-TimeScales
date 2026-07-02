"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pingouin as pg

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})

path = "../analyzed_data.pkl"
df_data = pd.read_pickle(path)


results = []

for exp in df_data.keys():

    pd1 = np.array(df_data[exp]["PD1"]["sdf_100_gauss"]) * 10
    pd2 = np.array(df_data[exp]["PD2"]["sdf_100_gauss"]) * 10
    lp  = np.array(df_data[exp]["LP"]["sdf_100_gauss"][1:]) * 10

    delay = np.array(df_data[exp]["intervals"]["LPPD1_delay"]) / 10
    pd1_period = np.array(df_data[exp]["intervals"]["PD1_period"]) / 10



    results.append({
        "exp": exp,
        "rho_PD1_PD2": spearmanr(pd1, pd2)[0],
        "rho_PD1_LP": spearmanr(pd1, lp)[0],
        "rho_PD1_delay": spearmanr(pd1, delay)[0],
        "rho_period_delay": spearmanr(pd1_period, delay)[0],
    })

df_corr = pd.DataFrame(results)

#DESCRIBE THE DF SHOW SOME STATS
df_corr.describe()





#VERTICAL BOXPLOTS

fig, axes = plt.subplots(4, 1, figsize=(4, 12))

data = [
    df_corr["rho_PD1_PD2"].dropna(),
    df_corr["rho_PD1_LP"].dropna(),
    df_corr["rho_PD1_delay"].dropna(),
    df_corr["rho_period_delay"].dropna(),
]

for ax, d in zip(axes, data):

    ax.boxplot(
        d,
        showmeans=True,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#d9d9d9", color="black"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=3),
    )

    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("ρ")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)

    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_box_aspect(1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("all_cycle_boxplots.svg")

print(np.mean(df_corr["rho_PD1_PD2"]))
print(np.mean(df_corr["rho_PD1_LP"]))
print(np.mean(df_corr["rho_PD1_delay"]))
print(np.mean(df_corr["rho_period_delay"]))


#BOXPLOTS
plt.figure(figsize=(10,5))

data_to_plot = [
    df_corr["rho_PD1_PD2"].dropna(),
    df_corr["rho_PD1_LP"].dropna(),
    df_corr["rho_PD1_delay"].dropna(),
    df_corr["rho_period_delay"].dropna(),
]

plt.boxplot(data_to_plot, showmeans=True, tick_labels=[
    "PD1 SDF-PD2 SDF",
    "PD1 SDF-LP SDF",
    "PD1 SDF-LPPD1delay",
    "period-LPPD1delay"
])

plt.ylabel("Spearman rho")
plt.title("Spearman correlation distributions across experiments")
plt.axhline(0, color="black", linewidth=0.8)
plt.show()



#HISTOGRAMS

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes = axes.ravel()
labels = ["PD1 SDF-PD2 SDF", "PD1 SDF-LP SDF", "PD1 SDF-LPPD1delay", "period-LPPD1delay"]
cols = ["rho_PD1_PD2", "rho_PD1_LP", "rho_PD1_delay", "rho_period_delay"]

for ax, label, col in zip(axes, labels, cols):
    ax.hist(df_corr[col].dropna(), bins=15, alpha=0.8)
    ax.set_title(label)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.show()