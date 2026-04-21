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

exps_to_ignore = []

metric_specs = [
    
    ("euclid_burst_mean", ["sync", "euclid_by_burst"], np.nanmean),
    ("euclid_cycle_mean", ["sync", "euclid_by_cycle"], np.nanmean),
    ("vpd_mean", ["sync", "vpd"], np.nanmean),

    ("PD1_sdf_100ms_mean", ["PD1", "sdf_100ms"], np.nanmean),
    ("PD2_sdf_100ms_mean", ["PD2", "sdf_100ms"], np.nanmean),
    ("LP_sdf_100ms_mean",  ["LP",  "sdf_100ms"], np.nanmean),

    ("PD1_avg_ISIs_mean", ["PD1", "avg_ISIs"], np.nanmean),
    ("PD2_avg_ISIs_mean", ["PD2", "avg_ISIs"], np.nanmean),
    ("LP_avg_ISIs_mean",  ["LP",  "avg_ISIs"], np.nanmean),

    #("PD1_period_cv",   ["PD1", "period_coefvar"],   np.nanmean),    
    #("PD2_period_cv",   ["PD2", "period_coefvar"],   np.nanmean),
    #("LP_period_cv",    ["LP",  "period_coefvar"],   np.nanmean),

    #("PD1_hyperpol_cv", ["PD1", "hyperpol_coefvar"], np.nanmean),
    #("PD2_hyperpol_cv", ["PD2", "hyperpol_coefvar"], np.nanmean),
    #("LP_hyperpol_cv",  ["LP",  "hyperpol_coefvar"], np.nanmean),

    #("PD1_burst_cv",    ["PD1", "burst_coefvar"],    np.nanmean),
    #("PD2_burst_cv",    ["PD2", "burst_coefvar"],    np.nanmean),
    #("LP_burst_cv",     ["LP",  "burst_coefvar"],    np.nanmean),


    ("PD1_period_cv",   ["intervals", "PD1_period"],   np.nanmean),
    ("PD2_period_cv",   ["intervals", "PD2_period"],   np.nanmean),
    ("LP_period_cv",   ["intervals", "LP_period"],   np.nanmean),

    ("PD1_hyperpol_cv",   ["intervals", "PD1_hyperpolarization"],   np.nanmean),
    ("PD2_hyperpol_cv",   ["intervals", "PD2_hyperpolarization"],   np.nanmean),
    ("LP_hyperpol_cv",   ["intervals", "LP_hyperpolarization"],   np.nanmean),

    ("PD1_burst_cv",   ["intervals", "PD1_burst"],   np.nanmean),
    ("PD2_burst_cv",   ["intervals", "PD2_burst"],   np.nanmean),
    ("LP_burst_cv",   ["intervals", "LP_burst"],   np.nanmean),


]

rows = []

rows = []
for exp_name, expdata in df_data.items():
    if exp_name in exps_to_ignore:
        continue
    for metric_name, path, func in metric_specs:
        arr = expdata
        for level in path:
            arr = arr[level]
        arr = np.asarray(arr, dtype=np.float64)
        # Ignore NaNs when computing CV
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            cv = np.nan
        else:
            cv = np.std(arr) / np.mean(arr)
        rows.append({
            "exp": exp_name,
            "metric": metric_name,
            "value": cv*100
        })

    # LPPD1 invariant
    A = np.asarray(expdata["intervals"]["LPPD1_delay"])
    B = np.asarray(expdata["intervals"]["PD1_period"])

    res = stats.linregress(A, B)

    rows.extend([
        {
            "exp": exp_name,
            "metric": "LPPD1_delay_r2",
            "value": res.rvalue ** 2
        }
    ])


    # LPPD2 invariant
    A = np.asarray(expdata["intervals"]["LPPD2_delay"])
    B = np.asarray(expdata["intervals"]["PD2_period"])

    res = stats.linregress(A, B)

    rows.extend([
        {
            "exp": exp_name,
            "metric": "LPPD2_delay_r2",
            "value": res.rvalue ** 2
        }
    ])
    
    # PD1LP invariant
    A = np.asarray(expdata["intervals"]["PD1LP_delay"])
    B = np.asarray(expdata["intervals"]["PD1_period"])

    res = stats.linregress(A, B)

    rows.extend([
        {
            "exp": exp_name,
            "metric": "PD1LP_delay_r2",
            "value": res.rvalue ** 2
        }
    ])


    
    # PD2LP invariant
    A = np.asarray(expdata["intervals"]["PD2LP_delay"])
    B = np.asarray(expdata["intervals"]["PD2_period"])

    res = stats.linregress(A, B)

    rows.extend([
        {
            "exp": exp_name,
            "metric": "PD2LP_delay_r2",
            "value": res.rvalue ** 2
        }
    ])


df_all_metrics = pd.DataFrame(rows)

# Pivot to wide format: one row per exp, one column per metric
df_wide = df_all_metrics.pivot_table(
    index="exp",
    columns="metric",
    values="value",
    sort=False
).reset_index()


metric_groups = {
    "Synchronization": [
        "vpd_mean",
        "euclid_burst_mean",
        "euclid_cycle_mean",
    ],
    "Excitability": [
        "PD1_sdf_100ms_mean",
        "PD2_sdf_100ms_mean",
        "LP_sdf_100ms_mean",
        "PD1_avg_ISIs_mean",
        "PD2_avg_ISIs_mean",
        "LP_avg_ISIs_mean",
    ],
    "Variability": [
        "PD1_period_cv",
        "PD2_period_cv",
        "LP_period_cv",
        "PD1_hyperpol_cv",
        "PD2_hyperpol_cv",
        "LP_hyperpol_cv",
        "PD1_burst_cv",
        "PD2_burst_cv",
        "LP_burst_cv",
    ],
    "Dynamical invariants": [
        "LPPD1_delay_r2",
        "LPPD2_delay_r2",
        "PD1LP_delay_r2",
        "PD2LP_delay_r2"
    ]
}


metric_display_names = {
    # --- Synchronization ---
    "vpd_mean": "VPD",
    "euclid_burst_mean": "EB",
    "euclid_cycle_mean": "EC",

    # --- Excitability ---
    "PD1_sdf_100ms_mean": "PD1 SDF",
    "PD2_sdf_100ms_mean": "PD2 SDF",
    "LP_sdf_100ms_mean":  "LP SDF",


    "PD1_avg_ISIs_mean": r"PD1 $\overline{ISI}$",
    "PD2_avg_ISIs_mean": r"PD2 $\overline{ISI}$",
    "LP_avg_ISIs_mean":  r"LP $\overline{ISI}$",

    # --- Variability ---
    "PD1_period_cv":   "PD1 period",
    "PD2_period_cv":   "PD2 period",
    "LP_period_cv":    "LP period",

    "PD1_hyperpol_cv": "PD1 hyperpol",
    "PD2_hyperpol_cv": "PD2 hyperpol",
    "LP_hyperpol_cv":  "LP hyperpol",

    "PD1_burst_cv":    "PD1 burst",
    "PD2_burst_cv":    "PD2 burst",
    "LP_burst_cv":     "LP burst",

    # --- Dynamical invariants ---
    "LPPD1_delay_r2":        "LPPD1delay R²",
    "LPPD2_delay_r2":        "LPPD2delay R²",

    "PD1LP_delay_r2":        "PD1LPdelay R²",
    "PD2LP_delay_r2":        "PD2LPdelay R²",
}



# ----------------------------
# Setup: metric groups & colors
# ----------------------------

group_order = [
    "Synchronization",
    "Excitability",
    "Variability",
]

group_colors = {
    "Synchronization": "#1f42b4",
    "Excitability": "#19ac31",
    "Variability": "#fd7600",
}



custom_metric_order = [
    "euclid_cycle_mean", "vpd_mean", "euclid_burst_mean", "LP_sdf_100ms_mean",
    "PD1_sdf_100ms_mean", "PD2_sdf_100ms_mean","LP_avg_ISIs_mean", "PD1_avg_ISIs_mean", "PD2_avg_ISIs_mean",
    "LP_period_cv", "PD1_period_cv", "PD2_period_cv",
    "LP_hyperpol_cv", "PD1_hyperpol_cv", "PD2_hyperpol_cv",
    "LP_burst_cv", "PD1_burst_cv", "PD2_burst_cv",
]



# Select metrics for boxplot
cv_metrics = [m for m, _, _ in metric_specs]  # only metrics where CV was computed
data_to_plot = [df_wide[m].dropna().values for m in cv_metrics]



# Prepare positions for boxes
positions = []
current_pos = 1
group_gap = 1  # extra gap between groups
box_gap = 1  # space between boxes within a group

data_to_plot_ordered = []
labels_gap = []

for group_name in group_order:
    metrics_in_group = [m for m in cv_metrics if m in metric_groups[group_name]]
    for m in metrics_in_group:
        data_to_plot_ordered.append(df_wide[m].dropna().values)
        positions.append(current_pos)
        labels_gap.append(metric_display_names.get(m, m))
        current_pos += box_gap
    current_pos += group_gap  # extra space after each group

# Boxplot with positions
plt.figure(figsize=(18,7))
bplot = plt.boxplot(
    data_to_plot_ordered,
    patch_artist=True,
    positions=positions,
    showmeans=True,
    showfliers=False,
    meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=7),
    medianprops=dict(color='black')
)

# Color boxes by group
group_colors_list = []
for label in labels_gap:
    for group_name, metrics in metric_groups.items():
        for m in metrics:
            if metric_display_names.get(m, m) == label:
                group_colors_list.append(group_colors[group_name])
                break
        else:
            continue
        break

for patch, color in zip(bplot['boxes'], group_colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Set x-ticks at box positions
print(positions)
plt.xticks(ticks=positions, labels=labels_gap, rotation=60, ha="center")
plt.ylabel("Coefficient of Variation (CV)")


plt.tight_layout()


# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("Boxplots_no_invariant.svg")

