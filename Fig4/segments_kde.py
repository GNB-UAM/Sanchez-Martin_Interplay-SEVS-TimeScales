"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})
chunk_size = 50

path = "../analyzed_data.pkl"
df_data = pd.read_pickle(path)


metric_specs = [
    # -------------------------
    # Single metrics
    # -------------------------
    dict(name="VPD", paths=["sync", "vpd"], reducer=np.nanmean),
    dict(name="EB", paths=["sync", "euclid_by_burst"], reducer=np.nanmean),
    dict(name="EC", paths=["sync", "euclid_by_cycle"], reducer=np.nanmean),
    dict(name="PD1 SDF", paths=["PD1", "sdf_100_gauss"], reducer=np.nanmean),
    dict(name="PD2 SDF", paths=["PD2", "sdf_100_gauss"], reducer=np.nanmean),
    dict(name="LP SDF", paths=["LP", "sdf_100_gauss"], reducer=np.nanmean),
    dict(name="PD1 spk/burst", paths=["PD1", "spikes_per_burst"], reducer=np.nanmean),
    dict(name="PD2 spk/burst", paths=["PD2", "spikes_per_burst"], reducer=np.nanmean),
    dict(name="LP spk/burst", paths=["LP", "spikes_per_burst"], reducer=np.nanmean),
    dict(name=r"PD1 $\overline{ISI}$", paths=["PD1", "avg_ISIs"], reducer=np.nanmean),
    dict(name=r"PD2 $\overline{ISI}$", paths=["PD2", "avg_ISIs"], reducer=np.nanmean),
    dict(name=r"LP $\overline{ISI}$", paths=["LP", "avg_ISIs"], reducer=np.nanmean),
    #dict(name="PD1 period CV", paths=["PD1", "period_coefvar"], reducer=np.nanmean),
    #dict(name="PD2 period CV", paths=["PD2", "period_coefvar"], reducer=np.nanmean),
    #dict(name="LP period CV", paths=["LP", "period_coefvar"], reducer=np.nanmean),
    #dict(name="PD1 hyperpol CV", paths=["PD1", "hyperpol_coefvar"], reducer=np.nanmean),
    #dict(name="PD2 hyperpol CV", paths=["PD2", "hyperpol_coefvar"], reducer=np.nanmean),
    #dict(name="LP hyperpol CV", paths=["LP", "hyperpol_coefvar"], reducer=np.nanmean),
    #dict(name="PD1 burst CV", paths=["PD1", "burst_coefvar"], reducer=np.nanmean),
    #dict(name="PD2 burst CV", paths=["PD2", "burst_coefvar"], reducer=np.nanmean),
    #dict(name="LP burst CV", paths=["LP", "burst_coefvar"], reducer=np.nanmean),

    dict(name="PD1 period CV", paths=["intervals", "PD1_period"], reducer=stats.variation),
    dict(name="PD2 period CV", paths=["intervals", "PD2_period"], reducer=stats.variation),
    dict(name="LP period CV", paths=["intervals", "LP_period"], reducer=stats.variation),
    dict(name="PD1 hyperpol CV", paths=["intervals", "PD1_hyperpolarization"], reducer=stats.variation),
    dict(name="PD2 hyperpol CV", paths=["intervals", "PD2_hyperpolarization"], reducer=stats.variation),
    dict(name="LP hyperpol CV", paths=["intervals", "LP_hyperpolarization"], reducer=stats.variation),
    dict(name="PD1 burst CV", paths=["intervals", "PD1_burst"], reducer=stats.variation),
    dict(name="PD2 burst CV", paths=["intervals", "PD2_burst"], reducer=stats.variation),
    dict(name="LP burst CV", paths=["intervals", "LP_burst"], reducer=stats.variation),
    
    # -------------------------
    # Binary / invariant metrics
    # -------------------------
    dict(
        name=rf"LPPD1 delay invariant $R^2$",
        paths=[
            ["intervals", "LPPD1_delay"],
            ["intervals", "PD1_period"],
        ],
        reducer="pearson_r2"
    ),
    dict(
        name="PD1LP_delay_invariant",
        paths=[
            ["intervals", "PD1LP_delay"],
            ["intervals", "PD1LP_interval"],
        ],
        reducer="pearson_r2"
    ),
]




def compute_chunked_metrics(df_data, metric_specs, chunk_size=chunk_size, exps_to_ignore=None):
    if exps_to_ignore is None:
        exps_to_ignore = []

    def resolve_paths(expdata, paths, reducer):
        if reducer == "pearson_r2":
            arr1 = expdata
            arr2 = expdata
            for p in paths[0]:
                arr1 = arr1[p]
            for p in paths[1]:
                arr2 = arr2[p]
            return np.asarray(arr1), np.asarray(arr2)
        else:
            arr = expdata
            for p in paths:
                arr = arr[p]
            return np.asarray(arr)

    def pearson_r2_vec(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        r = np.corrcoef(a[mask], b[mask])[0, 1]
        return r * r

    all_chunk_dfs = []

    for exp_name, expdata in df_data.items():
        if exp_name in exps_to_ignore:
            continue

        # ---- Resolve arrays once ----
        arrays_unary = {}
        arrays_binary = {}

        for spec in metric_specs:
            if spec["reducer"] == "pearson_r2":
                a, b = resolve_paths(expdata, spec["paths"], "pearson_r2")
                arrays_binary[spec["name"]] = (a, b)
            else:
                arr = resolve_paths(expdata, spec["paths"], spec["reducer"])
                arrays_unary[spec["name"]] = arr

        # ---- Compute chunks ----
        df_chunked = {"exp": [], "chunk": []}

        for name in arrays_unary:
            df_chunked[name] = []
        for name in arrays_binary:
            df_chunked[name] = []

        max_chunks = 0
        for arr in arrays_unary.values():
            max_chunks = max(max_chunks, len(arr) // chunk_size)
        for a, b in arrays_binary.values():
            max_chunks = max(max_chunks, min(len(a), len(b)) // chunk_size)

        for i in range(max_chunks):
            df_chunked["exp"].append(exp_name)
            df_chunked["chunk"].append(i)

            for name, arr in arrays_unary.items():
                reducer = next(s["reducer"] for s in metric_specs if s["name"] == name)
                if i < len(arr) // chunk_size:
                    chunk = arr[i * chunk_size:(i + 1) * chunk_size]
                    df_chunked[name].append(reducer(chunk))
                else:
                    df_chunked[name].append(np.nan)

            for name, (a, b) in arrays_binary.items():
                if i < min(len(a), len(b)) // chunk_size:
                    chunk_a = a[i * chunk_size:(i + 1) * chunk_size]
                    chunk_b = b[i * chunk_size:(i + 1) * chunk_size]
                    df_chunked[name].append(pearson_r2_vec(chunk_a, chunk_b))
                else:
                    df_chunked[name].append(np.nan)

        all_chunk_dfs.append(pd.DataFrame(df_chunked))

    if len(all_chunk_dfs) == 0:
        return pd.DataFrame()

    return pd.concat(all_chunk_dfs, ignore_index=True)


df_chunked = compute_chunked_metrics(
    df_data,
    metric_specs,
    chunk_size=chunk_size
)


#Convert sdf from 100ms⁻¹ to s⁻¹
df_chunked["PD1 SDF"]*= 10
df_chunked["PD2 SDF"]*= 10
df_chunked["LP SDF"]*= 10

df_plot = df_chunked.copy()

metrics_cols = df_plot.columns.difference(["exp", "chunk"])
metrics = metrics_cols.tolist()


def kde_colormesh(
    x, y,
    gridsize=200,
    bandwidth=None,
    cmap=None,
    vmin=None,
    normalize=False,
    x_label="x",
    y_label="y",
    ax=None,
    add_colorbar=True,
    pad_frac=0.1,
):
   

    x = np.asarray(x)
    y = np.asarray(y)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Colormap
    if cmap is None:
        cmap = "hot"

    else:
        cmap = cmap.copy()
        cmap.set_under("white")

    # KDE
    xy = np.vstack([x, y])
    kde = stats.gaussian_kde(xy)

    if bandwidth is not None:
        kde.set_bandwidth(kde.factor * bandwidth)

    # Grid with padding
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_pad = (x_max - x_min) * pad_frac
    y_pad = (y_max - y_min) * pad_frac

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, gridsize),
        np.linspace(y_min - y_pad, y_max + y_pad, gridsize)
    )

    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)

    if normalize:
        density /= density.max()

    # Main density plot
    mesh = ax.pcolormesh(
        xx, yy, density,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        rasterized=True
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_box_aspect(1)

    # KDE colorbar
    if add_colorbar:
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            label="KDE" + (" (normalized)" if normalize else "")
        )

    return ax, mesh


def get_xy_clean(df, metric_x, metric_y):
    x = df[metric_x].to_numpy()
    y = df[metric_y].to_numpy()

    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

plot_dict = {    
    "LP_sdf vs PD1_sdf": ("LP SDF",  "PD1 SDF"),
    "PD1_avg_ISIs vs PD2_avg_ISIs": (r"PD1 $\overline{ISI}$", r"PD2 $\overline{ISI}$"),
    "PD1_period vs PD1_hyperpolarization": ("PD1 period CV", "PD1 hyperpol CV"),
    "PD1_burst vs PD2_burst": ("PD1 burst CV", "PD2 burst CV"),
    #"vpd vs PD1_sdf": ("VPD", "PD1 SDF"),
    "Euclid_cycle vs PD1_sdf": ("PD1 SDF", "EC"),
    "LP_period vs PD1_sdf": ("PD1 SDF", "LP period CV"),
    "PD1_period vs LPPD1_delay": ("PD1 period CV", rf"LPPD1 delay invariant $R^2$"),
    "LPPD1_delay vs PD1_sdf ": ("PD1 SDF", rf"LPPD1 delay invariant $R^2$")
    
}

label_dict = {
    "LP SDF": "LP SDF (s⁻¹)",
    "PD1 SDF": "PD1 SDF (s⁻¹)",
    "PD1 $\overline{ISI}$": r"PD1 $\overline{ISI}$ (ms)",
    "PD2 $\overline{ISI}$": r"PD2 $\overline{ISI}$ (ms)",
    "PD1 period CV": "PD1 period CV",
    "PD1 hyperpol CV": "PD1 hyperpolarization CV",
    "PD1 burst CV": "PD1 burst CV",
    "PD2 burst CV": "PD2 burst CV",
    "VPD": "VPD",
    "EC": "EC (mV)",
    "LP period CV": "LP period CV",
    "LPPD1 delay invariant $R^2$": r"LPPD1 delay invariant $R^2$",
}


n_plots = len(plot_dict)

# Define subplot grid
n_cols = 4
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(6*n_cols, 5*n_rows),
)
axes = axes.flatten() #flatten for easy indexing

for idx, (key, (col_x, col_y)) in enumerate(plot_dict.items()):

    ax = axes[idx]

    x, y = get_xy_clean(df_plot, col_x, col_y)

    # --- filtering conditions ---
    if key == "PD1_period vs PD1_hyperpolarization":
        xmin, xmax = 0.01, 0.3
        ymin, ymax = 0.01, 0.3

        mask = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax)
        )

        x = x[mask]
        y = y[mask]


    if key == "PD1_period vs LPPD1_delay":
        xmin, xmax = 0, 0.5
        ymin, ymax = 0, 1

        mask = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax)
        )

        x = x[mask]
        y = y[mask]

    if key == "Euclid_cycle vs PD1_sdf":
        xmin, xmax = 0, 25
        ymin, ymax = 0, 100

        mask = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax)
        )

        x = x[mask]
        y = y[mask]

    if key == "LP_period vs PD1_sdf":
        xmin, xmax = 0, 25
        ymin, ymax = 0, 0.5

        mask = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax)
        )

        x = x[mask]
        y = y[mask]


    # --- Plot ---
    _, mesh = kde_colormesh(
        x, y,
        gridsize=200,
        vmin=0,
        x_label=col_x,
        y_label=col_y,
        normalize=True,
        add_colorbar=False,
        ax=ax
    )

    #ax.set_title(key)
    xlabel = label_dict.get(col_x, col_x)
    ylabel = label_dict.get(col_y, col_y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Hide unused subplots if any
for j in range(len(plot_dict), len(axes)):
    axes[j].set_visible(False)


for ax in axes:
    ax.set_box_aspect(1)

fig.subplots_adjust(wspace=0.03, hspace=0.3)
fig.savefig("segments_colormesh.svg")
