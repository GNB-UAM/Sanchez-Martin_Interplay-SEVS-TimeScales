"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from scipy.stats import linregress


#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures
plt.rcParams.update({
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../analyzed_data.pkl"
df_data = pd.read_pickle(path)

exps_to_ignore = []

metric_specs = [
    ("vpd_mean", ["sync", "vpd"], np.nanmean),
    ("euclid_burst_mean", ["sync", "euclid_by_burst"], np.nanmean),
    ("euclid_cycle_mean", ["sync", "euclid_by_cycle"], np.nanmean),

    ("PD1_sdf_100ms_mean", ["PD1", "sdf_100_gauss"], np.nanmean),
    ("PD2_sdf_100ms_mean", ["PD2", "sdf_100_gauss"], np.nanmean),
    ("LP_sdf_100ms_mean",  ["LP",  "sdf_100_gauss"], np.nanmean),

    ("PD1_spikes_per_burst_mean", ["PD1", "spikes_per_burst"], np.nanmean),
    ("PD2_spikes_per_burst_mean", ["PD2", "spikes_per_burst"], np.nanmean),
    ("LP_spikes_per_burst_mean",  ["LP",  "spikes_per_burst"], np.nanmean),

    ("PD1_avg_ISIs_mean", ["PD1", "avg_ISIs"], np.nanmean),
    ("PD2_avg_ISIs_mean", ["PD2", "avg_ISIs"], np.nanmean),
    ("LP_avg_ISIs_mean",  ["LP",  "avg_ISIs"], np.nanmean),

    ("PD1_period_cv",   ["intervals", "PD1_period"],   stats.variation),
    ("PD2_period_cv",   ["intervals", "PD2_period"],   stats.variation),
    ("LP_period_cv",   ["intervals", "LP_period"],   stats.variation),

    ("PD1_hyperpol_cv",   ["intervals", "PD1_hyperpolarization"],   stats.variation),
    ("PD2_hyperpol_cv",   ["intervals", "PD2_hyperpolarization"],   stats.variation),
    ("LP_hyperpol_cv",   ["intervals", "LP_hyperpolarization"],   stats.variation),

    ("PD1_burst_cv",   ["intervals", "PD1_burst"],   stats.variation),
    ("PD2_burst_cv",   ["intervals", "PD2_burst"],   stats.variation),
    ("LP_burst_cv",   ["intervals", "LP_burst"],   stats.variation),
    
]

rows = []

for exp_name, expdata in df_data.items():
    if exp_name in exps_to_ignore:
        continue
    #print(exp_name)
    # --- simple metrics ---
    for metric_name, path, func in metric_specs:
        arr = expdata
        for level in path:
            arr = arr[level]
        rows.append({
            "exp": exp_name,
            "metric": metric_name,
            "value": func(arr)
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
    
    # PD1LP invariant
    A = np.asarray(expdata["intervals"]["PD1LP_delay"])
    B = np.asarray(expdata["intervals"]["PD1LP_interval"])

    res = stats.linregress(A, B)

    rows.extend([
        {
            "exp": exp_name,
            "metric": "PD1LP_delay_r2",
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

#Convert sdf from 100ms⁻¹ to s⁻¹
df_wide["PD1_sdf_100ms_mean"]*= 10
df_wide["PD2_sdf_100ms_mean"]*= 10
df_wide["LP_sdf_100ms_mean"]*= 10


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
        "PD1_spikes_per_burst_mean",
        "PD2_spikes_per_burst_mean",
        "LP_spikes_per_burst_mean",
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
        "LPPD1_delay_slope",
        "LPPD1_delay_intercept",
        "PD1LP_delay_r2",
        "PD1LP_delay_slope",
        "PD1LP_delay_intercept",
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

    "PD1_spikes_per_burst_mean": "PD1 spikes/burst",
    "PD2_spikes_per_burst_mean": "PD2 spikes/burst",
    "LP_spikes_per_burst_mean":  "LP spikes/burst",

    "PD1_avg_ISIs_mean": "PD1 ISIs",
    "PD2_avg_ISIs_mean": "PD2 ISIs",
    "LP_avg_ISIs_mean":  "LP ISIs",

    # --- Variability ---
    "PD1_period_cv":   "PD1 period CV",
    "PD2_period_cv":   "PD2 period CV",
    "LP_period_cv":    "LP period CV",

    "PD1_hyperpol_cv": "PD1 hyperpol CV",
    "PD2_hyperpol_cv": "PD2 hyperpol CV",
    "LP_hyperpol_cv":  "LP hyperpol CV",

    "PD1_burst_cv":    "PD1 burst CV",
    "PD2_burst_cv":    "PD2 burst CV",
    "LP_burst_cv":     "LP burst CV",

    # --- Dynamical invariants ---
    "LPPD1_delay_r2":        "LP–PD1 r²",
    "LPPD1_delay_slope":     "LP–PD1 slope",
    "LPPD1_delay_intercept": "LP–PD1 intercept",

    "PD1LP_delay_r2":        "PD1–LP r²",
    "PD1LP_delay_slope":     "PD1–LP slope",
    "PD1LP_delay_intercept": "PD1–LP intercept",
}

exp_col = 'exp'

# Rank only metric columns
metric_cols = df_wide.columns.drop(exp_col)

ranked_df = df_wide.copy()
ranked_df[metric_cols] = (
    df_wide[metric_cols]
    .rank(axis=0, method='first', ascending=True)
    .astype(int)
)




def scatter_with_regression(x, y, ax=None,
                            xlabel='X', ylabel='Y',
                            title=None,
                            square=True, 
                            dotsize = 7,
                            plotline = True):


    x = np.asarray(x)
    y = np.asarray(y)

    """
    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]"""

    # Regression
    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r2 = result.rvalue ** 2
    #p_value = result.pvalue
    stderr = result.stderr
    
    rho, p_value = stats.spearmanr(x, y)
    #slope, intercept = np.polyfit(x, y, 1)
    #y_pred = slope * x + intercept
    #ss_res = np.sum((y - y_pred) ** 2)
    #ss_tot = np.sum((y - np.mean(y)) ** 2)
    #r2 = 1 - (ss_res / ss_tot)
    
    
    """rho, p = fast_spearman_perm(x,y)
    print(rho)
    print(p)"""


    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Scatter
    ax.scatter(x, y, s = dotsize)

    # Regression line
    x_line = np.sort(x)

    if plotline:
        ax.plot(x_line, slope * x_line + intercept, color='red')

    # Equation text
    #eq_text = f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r2:.2f}\n$p$-value = {p_value:.2e}'#With equation
    #eq_text = f'$R^2$ = {r2:.2f}\n$p$-value = {p_value:.2e}'#Without equation
    eq_text = f'$R^2$ = {r2:.2f}'#only R2
    ax.text(0.05, 0.95, eq_text,
            transform=ax.transAxes,
            verticalalignment='top')

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title is not None:
        ax.set_title(title)

    # Enforce square aspect ratio
    if square:
        ax.set_box_aspect(1)

    return fig, ax, slope, intercept, r2







fig, axs = plt.subplots(1, 4, figsize=(20, 6))
x1 = ranked_df["PD1_sdf_100ms_mean"]
y1 = ranked_df["PD2_sdf_100ms_mean"]

x2 = ranked_df["LPPD1_delay_r2"]
y2 = ranked_df["PD1_period_cv"]

x3 = ranked_df["PD1_sdf_100ms_mean"]
y3 = ranked_df["LPPD1_delay_r2"]

x4 = ranked_df["PD1_sdf_100ms_mean"]
y4 = ranked_df["PD1_period_cv"]

scatter_with_regression(x1, y1, ax=axs[0], xlabel = "PD1 SDF (rank)", ylabel="PD2 SDF (rank)", dotsize=30)
scatter_with_regression(x2, y2, ax=axs[1], xlabel = "PD1 period CV (rank)", ylabel= "LPPD1delay invariant $R^2$ (rank)", dotsize=30)
scatter_with_regression(x3, y3, ax=axs[2], xlabel = "PD1 SDF (rank)", ylabel= "LPPD1delay invariant $R^2$ (rank)", dotsize=30)
scatter_with_regression(x4, y4, ax=axs[3], xlabel = "PD1 SDF (rank)", ylabel= "PD1 period CV (rank)", dotsize=30)

plt.tight_layout()
plt.savefig("ranking_2Dplots.svg")
#plt.show()

