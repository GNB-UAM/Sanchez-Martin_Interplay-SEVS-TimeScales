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


def plot_cycle_by_cycle_metrics(exp, metrics_specs, start, end, figsize=(15, 2), marker='o', linewidth=1, markersize=3):
    n = len(metrics_specs)
    fig, axes = plt.subplots(n,1,sharex = True, figsize=(figsize[0], figsize[1] * n), constrained_layout = True)

    # Hide unnecessary spines
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Restore bottom axis
    last_ax = axes[-1]
    last_ax.tick_params(bottom=True, labelbottom=True, left=True)
    last_ax.spines['bottom'].set_visible(True)
    last_ax.spines['left'].set_visible(True)

    for i, metric in enumerate(metrics_specs):
        data = df_data[exp]
        
        # Traverse nested keys dynamically
        for key in metric["paths"]:
            data = data[key]

        # Apply transform if present
        if "transform" in metric:
            data = metric["transform"](data)
        
        axes[i].plot(data[start:end], color=metric["color"], label=metric["label"], marker=marker, linewidth=linewidth, markersize=markersize,)
        axes[i].set_ylabel(metric["label"].replace(" ", "\n"), rotation=0, labelpad=40)
        axes[i].grid(True, alpha=0.3, axis="x")

    axes[-1].set_xlabel("Cycle")
    return fig, axes


def transform_lp_sdf(x):
    y = pd.to_numeric(x, errors="coerce") * 10
    y = y[1:]
    return y

def transform_pds_sdf(x):
    y = pd.to_numeric(x, errors="coerce") * 10
    y = y.copy()
    return y

def transform_intervals(x):
    y = pd.to_numeric(x, errors="coerce") / 10
    y = y.copy()
    return y

metrics_specs = [
    dict(name="EC", paths=["sync", "euclid_by_cycle"], color = "#1f42b4", label = "EC (mV)"),
    dict(name="PD1 SDF", paths=["PD1", "sdf_100_gauss"], color = "#19ac31", label = "PD1 SDF (s⁻¹)", transform = transform_pds_sdf),
    dict(name="PD2 SDF", paths=["PD2", "sdf_100_gauss"], color = "#19ac31", label = "PD2 SDF (s⁻¹)", transform= transform_pds_sdf),
    dict(name="LP SDF", paths=["LP", "sdf_100_gauss"], color = "#19ac31", label = "LP SDF (s⁻¹)", transform= transform_lp_sdf),
    dict(name="PD1LP delay", paths=["intervals", "PD1LP_delay"], color = "#fd7600", label = "PD1LP delay (ms)", transform= transform_intervals),
    dict(name="LPPD1 delay", paths=["intervals", "LPPD1_delay"], color = "#fd7600", label = "LPPD1 delay (ms)", transform= transform_intervals),
    dict(name="PD1 period", paths=["intervals", "PD1_period"], color = "#fd7600", label = "PD1 period (ms)", transform= transform_intervals),
    ]


fig, axes = plot_cycle_by_cycle_metrics(exp = "12", metrics_specs = metrics_specs, start = 0, end= 100)
fig.savefig("cycle-by-cycle.svg")
