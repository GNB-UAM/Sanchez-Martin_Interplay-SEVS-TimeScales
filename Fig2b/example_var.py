"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures
plt.rcParams['svg.fonttype'] = 'none'

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../analyzed_data.pkl"
df_data = pd.read_pickle(path)


Extra1 = np.loadtxt("example_var_exp1.csv", usecols = (0), skiprows = 1, delimiter=',')
Intra1 = np.loadtxt("example_var_exp1.csv", usecols = (1), skiprows = 1, delimiter=',')

plt.close()
plt.figure(figsize=(12, 6))
plt.plot(Extra1*3, c=(0.15, 0.15, 0.15), linewidth = 1.2)
plt.plot(Intra1+1.5, c = "tab:blue", linewidth = 1.2)
plt.axis('off')
plt.savefig("example_var_exp1.svg")
plt.show()