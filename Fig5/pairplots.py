"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})

path = "../analyzed_data.pkl"
df_data = pd.read_pickle(path)


exp = "12"


plt.rcParams.update({'font.size': 20})

number_cycles = 1000


x0, y0 = np.array(df_data[exp]["PD1"]["sdf_100_gauss"][:number_cycles])*10, np.array(df_data[exp]["PD2"]["sdf_100_gauss"][:number_cycles])*10
x1, y1 = np.array(df_data[exp]["PD1"]["sdf_100_gauss"][:number_cycles])*10, np.array(df_data[exp]["LP"]["sdf_100_gauss"][1:number_cycles+1])*10
x2, y2 = np.array(df_data[exp]["PD1"]["sdf_100_gauss"][:number_cycles])*10, np.array(df_data[exp]["intervals"]["LPPD1_delay"][:number_cycles])/10
x3, y3 = np.array(df_data[exp]["intervals"]["PD1_period"][:number_cycles])/10, np.array(df_data[exp]["intervals"]["LPPD1_delay"][:number_cycles])/10


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 12), sharex=False)


axes[0].scatter(x0, y0, s = 5, c ="#555555")
axes[0].set_xlabel("PD1 SDF (s⁻¹)")
axes[0].set_ylabel("PD2 SDF (s⁻¹)")


axes[1].scatter(x1, y1, s = 5, c ="#555555")
axes[1].set_xlabel("PD1 SDF (s⁻¹)")
axes[1].set_ylabel("LP SDF (s⁻¹)")


axes[2].scatter(x2, y2, s = 5, c ="#555555")
axes[2].set_xlabel("PD1 SDF (s⁻¹)")
axes[2].set_ylabel("LPPD1 delay (ms)")



axes[3].scatter(x3, y3, s = 5, c ="#555555")
axes[3].set_xlabel("PD1 Period (ms)")
axes[3].set_ylabel("LPPD1 delay (ms)")


for ax in axes:
    ax.set_box_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plt.tight_layout()

plt.savefig("pairplots.svg")

