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

exp = df_data["6"]



x = exp["intervals"]["PD1_period"]
y1 = exp["intervals"]["LPPD1_delay"]
y2 = exp["intervals"]["PD1_burst"]


fig, ax = plt.subplots(figsize=(6, 6))

# Scatter
ax.scatter(x, y1, s=5, c = "red")
ax.scatter(x, y2, s=5, c = "black")

# Regression
reg1 = stats.linregress(x, y1)
reg2 = stats.linregress(x, y2)


print(len(x))

# Smooth x-range for lines
x_line = np.linspace(np.min(x), np.max(x), 200)

ax.plot(x_line, reg1.slope*x_line + reg1.intercept,
        label=f"LPPD delay (R²={reg1.rvalue**2:.2f})", c = "red", linewidth = 2)

ax.plot(x_line, reg2.slope*x_line + reg2.intercept,
        label=f"PD burst (R²={reg2.rvalue**2:.2f})", c="black", linewidth = 2)



ax.set_xticks([1000,1500,2000])
ax.set_yticks([0,500,1000,1500])

ax.set_xlabel("PD period (ms)")
ax.set_ylabel("Interval (ms)")

import matplotlib.ticker as ticker

"""# Apply thousands comma separator to the y-axis
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))"""


ax.legend(fontsize=15)

ax.set_box_aspect(1)

plt.tight_layout()
plt.savefig("example_invariant.svg")