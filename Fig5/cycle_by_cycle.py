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


def plot_cycle_metrics(
    df_cycles,
    metrics,
    exp,
    start,
    end,
    cycle_col=None,
    figsize=(15, 2),
    marker='o',
    linewidth=1,
    markersize=3,
    colors=None,
    labels=None
):

    # --- filter experiment ---
    df = df_cycles[df_cycles["exp"] == exp]

    if df.empty:
        raise ValueError(f"No data found for exp = {exp}")

    n = len(metrics)
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        figsize=(figsize[0], figsize[1] * n),
        constrained_layout=True
    )

    if n == 1:
        axes = [axes]

    # Hide unnecessary spines
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Restore bottom axis
    last_ax = axes[-1]
    last_ax.tick_params(bottom=True, labelbottom=True, left=True)
    last_ax.spines['bottom'].set_visible(True)
    last_ax.spines['left'].set_visible(True)

    x = df[cycle_col] if cycle_col is not None else df.index

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in df_cycles")

        # --- determine color ---
        if colors is None:
            color = None
        elif isinstance(colors, dict):
            color = colors.get(metric, None)
        else:
            color = colors[i]

        # --- determine label ---
        if labels is None:
            label = metric.replace(" ", "\n")
        elif isinstance(labels, dict):
            label = labels.get(metric, metric)
        else:
            label = labels[i]

        ax.plot(
            x[start:end],
            df[metric][start:end],
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            color=color
        )

        ax.set_ylabel(label.replace(" ", "\n"), rotation=0, labelpad=40)
        ax.grid(True, alpha=0.3, axis="x")

    axes[-1].set_xlabel("Cycle")

    return fig, axes


metric_specs = [
    # -------------------------
    # Single metrics
    # -------------------------
    dict(name="VPD", paths=["sync", "vpd"], reducer=np.nanmean),
    dict(name="EB", paths=["sync", "euclid_by_burst"], reducer=np.nanmean),
    dict(name="EC", paths=["sync", "euclid_by_cycle"], reducer=np.nanmean),
    dict(name="PD1 SDF", paths=["PD1", "sdf_100ms"], reducer=np.nanmean),
    dict(name="PD2 SDF", paths=["PD2", "sdf_100ms"], reducer=np.nanmean),
    dict(name="LP SDF", paths=["LP", "sdf_100ms"], reducer=np.nanmean),
    dict(name="PD1 spk/burst", paths=["PD1", "spikes_per_burst"], reducer=np.nanmean),
    dict(name="PD2 spk/burst", paths=["PD2", "spikes_per_burst"], reducer=np.nanmean),
    dict(name="LP spk/burst", paths=["LP", "spikes_per_burst"], reducer=np.nanmean),
    dict(name="PD1 avg ISIs", paths=["PD1", "avg_ISIs"], reducer=np.nanmean),
    dict(name="PD2 avg ISIs", paths=["PD2", "avg_ISIs"], reducer=np.nanmean),
    dict(name="LP avg ISIs", paths=["LP", "avg_ISIs"], reducer=np.nanmean),
    dict(name="PD1 period cv", paths=["PD1", "period_coefvar"], reducer=np.nanmean),
    dict(name="PD2 period cv", paths=["PD2", "period_coefvar"], reducer=np.nanmean),
    dict(name="LP period cv", paths=["LP", "period_coefvar"], reducer=np.nanmean),
    dict(name="PD1 hyperpol cv", paths=["PD1", "hyperpol_coefvar"], reducer=np.nanmean),
    dict(name="PD2 hyperpol cv", paths=["PD2", "hyperpol_coefvar"], reducer=np.nanmean),
    dict(name="LP hyperpol cv", paths=["LP", "hyperpol_coefvar"], reducer=np.nanmean),
    dict(name="PD1 burst cv", paths=["PD1", "burst_coefvar"], reducer=np.nanmean),
    dict(name="PD2 burst cv", paths=["PD2", "burst_coefvar"], reducer=np.nanmean),
    dict(name="LP burst cv", paths=["LP", "burst_coefvar"], reducer=np.nanmean),
    dict(name="LPPD1 delay", paths=["intervals", "LPPD1_delay"], reducer = lambda x: np.nanmean(x) / 10),
    dict(name="PD1LP delay", paths=["intervals", "PD1LP_delay"], reducer=lambda x: np.nanmean(x) / 10),
    dict(name="PD1 period", paths=["intervals", "PD1_period"], reducer = lambda x: np.nanmean(x) / 10),

]



def df_to_wide(
    df_data,
    metric_specs,
    compute_cycles=True,
    exps_to_ignore=None,
    fillna_cycles=None,
):

    if exps_to_ignore is None:
        exps_to_ignore = []

    df_wide_dict = {}


    def resolve_paths(expdata, paths):
        arr = expdata
        for p in paths:
            arr = arr[p]
        return np.asarray(arr)

    # -------------------- main loop --------------------
    for exp_name, expdata in df_data.items():
        if exp_name in exps_to_ignore:
            continue

        # --- resolve arrays ---
        arrays_unary = {}
        arrays_binary = {}
        for spec in metric_specs:
            reducer = spec["reducer"]
            arr = resolve_paths(expdata, spec["paths"])
            # apply reducer after extraction
            if callable(reducer):
                arr = np.asarray([reducer(x) for x in arr])

            arrays_unary[spec["name"]] = arr

        # --------------------COMPUTE CYCLES --------------------
        if compute_cycles:
            df_cycles_dict = {"exp": []}
            for name in arrays_unary.keys():
                df_cycles_dict[name] = []
            for name in arrays_binary.keys():
                df_cycles_dict[name] = []

            n_cycles = max(len(arr) for arr in arrays_unary.values()) if arrays_unary else 0
            for i in range(n_cycles):
                df_cycles_dict["exp"].append(exp_name)
                for name, arr in arrays_unary.items():
                    df_cycles_dict[name].append(arr[i] if i < len(arr) else (fillna_cycles if fillna_cycles is not None else np.nan))
                for name, (a,b) in arrays_binary.items():
                    df_cycles_dict[name].append(fillna_cycles if fillna_cycles is not None else np.nan)

            df_wide_dict.setdefault("cycles", []).append(
                pd.DataFrame({"cycle": np.arange(n_cycles), **df_cycles_dict})
            )
    # -------------------- concatenate per type --------------------
    for key in df_wide_dict.keys():
        df_wide_dict[key] = pd.concat(df_wide_dict[key], ignore_index=True)

    return df_wide_dict


df_dict = df_to_wide(
    df_data,
    metric_specs,
    compute_cycles=True,
    exps_to_ignore=None,
    fillna_cycles=0
)
df_cycles = df_dict["cycles"]

#Convert sdf from 100ms⁻¹ to s⁻¹
df_cycles["PD1 SDF"]*= 10
df_cycles["PD2 SDF"]*= 10
df_cycles["LP SDF"]*= 10


colors=["#1f42b4","#19ac31","#19ac31","#19ac31","#fd7600", "#fd7600", "#fd7600"]

custom_labels = {
        "VPD": "VPD",
        "EB": "EB (mV)",
        "EC": "EC (mV)",
        "PD1 SDF": "PD1 SDF (s⁻¹)",
        "PD2 SDF": "PD2 SDF (s⁻¹)",
        "LP SDF": "LP SDF (s⁻¹)",
        "PD1LP delay": "PD1LP delay (ms)",
        "LPPD1 delay":  "LPPD1 delay (ms)",
        "PD1 period": "PD1 period (ms)"
    }

fig, axes = plot_cycle_metrics(df_cycles, metrics=["EC", "PD1 SDF", "PD2 SDF", "LP SDF", "PD1LP delay", "LPPD1 delay", "PD1 period"], exp = "12", start = 0, end = 100, cycle_col = "cycle", colors = colors, labels = custom_labels)
fig.savefig("cycle-by-cycle.svg")

