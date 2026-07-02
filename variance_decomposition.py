"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures


#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "./analyzed_data.pkl"
df_data = pd.read_pickle(path)


all_LPPD1delay_ratios = []
all_LPPD2delay_ratios = []
all_LP_ratios = []
all_PD1_ratios = []
all_PD2_ratios = []
all_PD1LP_ratios = []
all_PD2LP_ratios = []
all_PD1_hyper_ratios = []
all_PD2_hyper_ratios = []
all_LP_hyper_ratios = []

total_var_period_var = []

all_sum_of_others = []

for exp in df_data.keys():


    PD1_period = df_data[exp]["intervals"]["PD1_period"]
    PD2_period = df_data[exp]["intervals"]["PD2_period"]
    PD1_hyper = df_data[exp]["intervals"]["PD1_hyperpolarization"]
    PD2_hyper = df_data[exp]["intervals"]["PD2_hyperpolarization"]
    PD1_burst =  df_data[exp]["intervals"]["PD1_burst"]
    PD2_burst =  df_data[exp]["intervals"]["PD2_burst"]

    LP_period = df_data[exp]["intervals"]["LP_period"]
    LP_hyper = df_data[exp]["intervals"]["LP_hyperpolarization"]
    LP_burst =  df_data[exp]["intervals"]["LP_burst"]


    LPPD1_delay = df_data[exp]["intervals"]["LPPD1_delay"]
    PD1LP_delay = df_data[exp]["intervals"]["PD1LP_delay"]

    LPPD2_delay = df_data[exp]["intervals"]["LPPD2_delay"]
    PD2LP_delay = df_data[exp]["intervals"]["PD2LP_delay"]




    PD1_hyper_ratios = (np.std(PD1_hyper)**2) / (np.std(PD1_period)**2)
    all_PD1_hyper_ratios.append(PD1_hyper_ratios)

    PD2_hyper_ratios = (np.std(PD2_hyper)**2) / (np.std(PD2_period)**2)
    all_PD2_hyper_ratios.append(PD2_hyper_ratios)

    LPPD1_delay_ratio = (np.std(LPPD1_delay)**2) / (np.std(PD1_period)**2)
    all_LPPD1delay_ratios.append(LPPD1_delay_ratio)

    LPPD2_delay_ratio = (np.std(LPPD2_delay)**2) / (np.std(PD2_period)**2)
    all_LPPD2delay_ratios.append(LPPD2_delay_ratio)

    LP_ratio = (np.std(LP_burst)**2) / (np.std(PD1_period)**2)
    all_LP_ratios.append(LP_ratio)

    PD1_ratio = (np.std(PD1_burst)**2) / (np.std(PD1_period)**2)
    all_PD1_ratios.append(PD1_ratio)

    PD2_ratio = (np.std(PD2_burst)**2) / (np.std(PD2_period)**2)
    all_PD2_ratios.append(PD2_ratio)

    PD1LP_delay_ratio = (np.std(PD1LP_delay)**2) / (np.std(PD1_period)**2)
    all_PD1LP_ratios.append(PD1LP_delay_ratio)

    PD2LP_delay_ratio = (np.std(PD2LP_delay)**2) / (np.std(PD2_period)**2)
    all_PD2LP_ratios.append(PD2LP_delay_ratio)

    sum_of_others = ((np.std(PD1LP_delay)**2) + (np.std(PD1_burst)**2) + (np.std(LP_burst)**2)) / (np.std(PD2_period)**2)
    all_sum_of_others.append(sum_of_others)

    # LPPD1 vs period regression
    # fit line y = ax + b
    x = np.array(PD1_period)
    y = np.array(LPPD1_delay)
    a, b = np.polyfit(x, y, 1)

    y_pred = a * x + b
    residuals = y - y_pred

    #plt.scatter(residuals, df_data[exp]["PD1"]["sdf_100_gauss"])
    #plt.scatter(residuals, y)
    #plt.show()


    regress = linregress(LPPD1_delay, PD1_period)
    
    plt.scatter(LPPD1_delay_ratio, regress.rvalue**2)
    print(LPPD1_delay_ratio)

    #Total variance vs period variance
    arrays = [LPPD1_delay, LP_burst, PD1_burst, PD1LP_delay]
    X = np.vstack(arrays)
    cov = np.cov(X)
    total_variance = np.sum(cov)

    total_var_period_var.append((np.std(PD1_period)**2)/total_variance)

    print(f"total variance: {total_variance}; period variance:  {(np.std(PD1_period)**2)}")
    





print(f"LPPD1 variance ratio across al exps: {np.mean(all_LPPD1delay_ratios):.2f}")

print(f"LPPD2 variance ratio across al exps: {np.mean(all_LPPD2delay_ratios):.2f}")

print(f"LP burst variance ratio across al exps: {np.mean(all_LP_ratios):.2f}")

print(f"PD1 burst variance ratio across al exps: {np.mean(all_PD1_ratios):.2f}")

print(f"PD2 burst variance ratio across al exps: {np.mean(all_PD2_ratios):.2f}")

print(f"PD1 hyper variance ratio across al exps: {np.mean(all_PD1_hyper_ratios):.2f}")

print(f"PD2 hyper variance ratio across al exps: {np.mean(all_PD2_hyper_ratios):.2f}")

print(f"PD1LP variance ratio across al exps: {np.mean(all_PD1LP_ratios):.2f}")

print(f"PD2LP variance ratio across al exps: {np.mean(all_PD2LP_ratios):.2f}")

print(f"Total variance % period variance: {np.mean(total_var_period_var):.4f}")

print(f"sum of others ratio period variance: {np.mean(all_sum_of_others):.2f}")

plt.show()
