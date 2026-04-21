"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures


#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../analysed_data.pkl"

df_data = pd.read_pickle(path)


example_burstPD1_exp1 = np.loadtxt("example_sync_exp1.csv", usecols = (0), skiprows = 1, delimiter=',')
example_burstPD2_exp1 = np.loadtxt("example_sync_exp1.csv", usecols = (1), skiprows = 1, delimiter=',')
example_burstPD1_exp5 = np.loadtxt("example_sync_exp5.csv", usecols = (0), skiprows = 1, delimiter=',')
example_burstPD2_exp5 = np.loadtxt("example_sync_exp5.csv", usecols = (1), skiprows = 1, delimiter=',')

plt.plot(example_burstPD1_exp1)
plt.plot(example_burstPD2_exp1)
plt.axis('off')
plt.text(0.5,0.3,  f"VPD = {df_data['1']['sync']['vpd'][0]:.2f}", fontsize=18, ha='center', va='center',transform=plt.gca().transAxes)
plt.text(0.5,0.2,  f"EB = {df_data['1']['sync']['euclid_by_burst'][0]:.2f}", fontsize=18, ha='center', va='center',transform=plt.gca().transAxes)
plt.savefig("example_vpd_1.svg")
plt.show()




plt.plot(example_burstPD1_exp5-2.3)#Vertical Offset
plt.plot(example_burstPD2_exp5)
plt.axis('off')
plt.text(0.5,0.3,  f"VPD = {df_data['5']['sync']['vpd'][0]:.2f}", fontsize=18, ha='center', va='center',transform=plt.gca().transAxes)
plt.text(0.5,0.2,  f"EB = {df_data['5']['sync']['euclid_by_burst'][0]:.2f}", fontsize=18, ha='center', va='center',transform=plt.gca().transAxes)
plt.savefig("example_vpd_5.svg")
plt.show()




