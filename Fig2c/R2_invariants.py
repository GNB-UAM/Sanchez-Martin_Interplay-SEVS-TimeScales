"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as stats


#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../analyzed_data.pkl"
df_data = pd.read_pickle(path)


exps_to_ignore_LPPDdelay = []#For testing
exps_to_ignore_PDLPdelay= []#For testing



LPPD1delay_exps = []
LPPD2delay_exps = []
PD1LPdelay_exps = []
PD2LPdelay_exps = []

for exp in df_data.keys():
    if exp not in exps_to_ignore_LPPDdelay:
        x = df_data[exp]["intervals"]["PD1_period"]
        y = df_data[exp]["intervals"]["LPPD1_delay"]
        reg = stats.linregress(x,y)
        LPPD1delay_exps.append(reg.rvalue**2)

        x = df_data[exp]["intervals"]["PD2_period"]
        y = df_data[exp]["intervals"]["LPPD2_delay"]
        reg = stats.linregress(x,y)
        LPPD2delay_exps.append(reg.rvalue**2)


    if exp not in exps_to_ignore_PDLPdelay:
        x = df_data[exp]["intervals"]["PD1_period"]
        y = df_data[exp]["intervals"]["PD1LP_delay"]
        reg = stats.linregress(x,y)
        PD1LPdelay_exps.append(reg.rvalue**2)

        x = df_data[exp]["intervals"]["PD2_period"]
        y = df_data[exp]["intervals"]["PD2LP_delay"]
        reg = stats.linregress(x,y)
        PD2LPdelay_exps.append(reg.rvalue**2)


plt.figure(figsize=(6, 6))

boxprops = dict(facecolor= "#df1212", alpha=0.8)
medianprops = dict(color='black', linewidth=2)

# Plot each boxplot
plt.boxplot(LPPD1delay_exps, positions=[1], widths=0.4,
            patch_artist=True, boxprops=boxprops, medianprops=medianprops, showfliers=True)

plt.boxplot(LPPD2delay_exps, positions=[2], widths=0.4,
            patch_artist=True, boxprops=boxprops, medianprops=medianprops, showfliers=True)

plt.boxplot(PD1LPdelay_exps, positions=[4], widths=0.4,
            patch_artist=True, boxprops=boxprops, medianprops=medianprops, showfliers=True)

plt.boxplot(PD2LPdelay_exps, positions=[5], widths=0.4,
            patch_artist=True, boxprops=boxprops, medianprops=medianprops, showfliers=True)

plt.ylabel("R²")
plt.xticks([1,2,4,5], ["LPPD1 delay", "LPPD2 delay", "PD1LP delay", "PD2LP delay"], rotation=60, ha="right")


# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("boxplots_invariants.svg")


#VIOLION PLOTS
"""

plt.close()
# LPPD1
v1 = plt.violinplot(LPPD1delay_exps, positions=[1], widths=0.6, showmeans=False, showmedians=True, showextrema=True)

# LPPD2
v2 = plt.violinplot(LPPD2delay_exps, positions=[2], widths=0.6, showmeans=False, showmedians=True, showextrema=True)

# PD1LP
v3 = plt.violinplot(PD1LPdelay_exps, positions=[4], widths=0.6, showmeans=False, showmedians=True, showextrema=True)

# PD2LP
v4 = plt.violinplot(PD2LPdelay_exps, positions=[5], widths=0.6, showmeans=False, showmedians=True, showextrema=True)


for v in [v1, v2, v3, v4]:
    for body in v['bodies']:
        body.set_facecolor('red')
        body.set_edgecolor('black')
        body.set_alpha(0.7)

        # inner lines → make them black
    for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if part in v:
            v[part].set_edgecolor('black')
            v[part].set_linewidth(1.5)


plt.ylabel("R²")
plt.xticks([1,2,4,5], ["LPPD1", "LPPD2", "PD1LP", "PD2LP"], rotation=45, ha="right")

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("violin_invariants.svg")"""
