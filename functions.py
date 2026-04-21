"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pandas as pd
import scipy as scp
#import scipy.stats
import math as math
from functools import reduce
import operator
from scipy import stats




# ----------------------------
#  MATH FUNCTIONS
# ----------------------------

#The difference of this function to scp.signal.windows.gaussian is normalizing to area instead of max (1)
def gauss(n=700,sigma=100):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]


# ----------------------------
#  EXCITABILITY METRICS FUNCTIONS
# ----------------------------


def spike_density_function(neuron_spikes, neuron_beg, fs=10000, window_ms=100, window_type="gaussian"):
    spikes_int = np.array([int(i) for i in neuron_spikes * 10])    
    spike_train = np.zeros(int(neuron_spikes[-1]) * 10 + 10)
    spike_train[spikes_int] = 1
    window_size = int(fs * (window_ms / 1000.0))

    # choose gaussian or squared window
    if window_type == "gaussian":
        window = scp.signal.windows.gaussian(window_size, std=window_size / 6)
    elif window_type == "squared":
        window = np.ones(window_size)

    sdf = scp.signal.convolve(spike_train, window, mode='same')
    
    sdf_per_cycle = []
    for i in range(len(neuron_beg)-1):
        sdf_per_cycle.append((np.mean(sdf[int(neuron_beg[i]*10):int(neuron_beg[i+1]*10)])))
    return sdf_per_cycle, sdf


def spike_density_function_burst(neuron_spikes, neuron_beg, neuron_end, fs=10000, window_ms=100, window_type="gaussian"):
    spikes_int = np.array([int(i) for i in neuron_spikes * 10])
    spike_train = np.zeros(int(neuron_spikes[-1]) * 10 + 10)
    spike_train[spikes_int] = 1
    window_size = int(fs * (window_ms / 1000.0))

    # choose gaussian or squared window
    if window_type == "gaussian":
        window = scp.signal.windows.gaussian(window_size, std=window_size / 6)
    elif window_type == "squared":
        window = np.ones(window_size)

    sdf = scp.signal.convolve(spike_train, window, mode='same')

    sdf_per_burst = []
    for i in range(len(neuron_beg)-1):
        sdf_per_burst.append((np.mean(sdf[int(neuron_beg[i]*10):int(neuron_end[i]*10)])))
    return sdf_per_burst, sdf



def spikes_per_time(neuron_beg, neuron_end, relation):
    spikes_per_time = []  
    for burst_num in range(1, len(neuron_beg)):
        burst_start = neuron_beg[burst_num - 1]
        cycle_end = neuron_beg[burst_num]        
        spikes_in_cycle = [spike_time for burst, spike_time in relation if burst == burst_num and burst_start <= spike_time < cycle_end]
        cycle_duration = cycle_end - burst_start

        if cycle_duration > 0:
            spike_rate = len(spikes_in_cycle) / cycle_duration
            spikes_per_time.append(spike_rate)
        else:
            spikes_per_time.append(0)
    
    return spikes_per_time


def ISIs(neuron_spikes, interburst_threshold = 100):
    neuron_ISIs = []
    for i in range(len(neuron_spikes)-1):
        if neuron_spikes[i+1]-neuron_spikes[i] < interburst_threshold:
            neuron_ISIs.append(neuron_spikes[i+1]-neuron_spikes[i])
    return neuron_ISIs


def IBIs(neuron_beg, neuron_end, interburst_threshold = 100):
    neuron_IBIs = []
    for i in range(len(neuron_beg)-1):
        if neuron_beg[i+1]-neuron_end[i] > interburst_threshold:
            neuron_IBIs.append(neuron_beg[i+1]-neuron_end[i])
    return neuron_IBIs


def burst_spike_relation(neuron_beg,neuron_end, neuron_spikes, offset = 80):
	detected_only_neuron_spikes = []
	burst_number_for_this_spike = 1
	burst_spike_relation = np.array(())
	for i in range(len(neuron_beg)):
		temp = np.extract(np.logical_and(neuron_spikes < (neuron_end[i]+offset), neuron_spikes > (neuron_beg[i]-offset)), neuron_spikes)
		burst_spike_relation = np.append(burst_spike_relation, [burst_number_for_this_spike] * temp.size)
		detected_only_neuron_spikes = np.append(detected_only_neuron_spikes, temp)
		burst_number_for_this_spike += 1
	neuron_relation = np.column_stack((burst_spike_relation, detected_only_neuron_spikes))
	return neuron_relation


def avg_ISIs(neuron_beg, neuron_relation):
    neuron_avg_ISIs = []
    for i in range(1,len(neuron_beg)+1):
        this_cycle_spike_arrays = neuron_relation[neuron_relation[:, 0] == i][:,1]
        this_cycle_ISIs = (np.roll(this_cycle_spike_arrays, -1)-this_cycle_spike_arrays)[:-1]
        this_cycle_ISIs_avg = np.mean(this_cycle_ISIs)
        neuron_avg_ISIs.append(this_cycle_ISIs_avg)
    return neuron_avg_ISIs


def spike_number_per_burst(neuron_beg,neuron_relation):
    spikes_per_burst = []
    for i in range(1, len(neuron_beg)+1):
        spikes_per_burst.append(np.count_nonzero(neuron_relation == i))
    return spikes_per_burst


# ----------------------------
#  SYNCHRONIZATION METRICS FUNCTIONS
# ----------------------------


def VPD(tli,tlj,cost):
        
    nspi=len(tli);
    nspj=len(tlj);
    
    if cost==0:
        d=abs(nspi-nspj);
        return d 
    elif cost==np.inf:
        d=nspi+nspj;
        return d
    scr=np.zeros((nspi+1,nspj+1));
    
    
    scr[:,0]=np.transpose(np.arange(nspi+1));
    scr[0,:]=np.arange(nspj+1);
    
    if nspi and nspj:
        for i in range(1, int(nspi+1)):
            for j in range(1, int(nspj+1)):
                temp=np.concatenate(([scr[i-1,j]+1], [scr[i,j-1]+1], [scr[i-1,j-1]+cost*abs(tli[i-1]-tlj[j-1])]), axis = 0); 
                scr[i,j]=temp.min(0);
    d=scr[nspi, nspj];
    return d

def victor_purpura_distance(neuron1_relation, neuron2_relation, cost = 0.03):#Adapted directly from C++ implementation
	min=neuron1_relation[0,0];max=neuron1_relation[len(neuron1_relation)-1,0];
	vpd = np.array([]);
	for i in range(int(min), int(max)+1):
		Bt = neuron1_relation[neuron1_relation[:,0]==i,1]; 
		Btt = neuron2_relation[neuron2_relation[:,0]==i,1]; 
		d = VPD(Bt,Btt,cost);
		vpd = np.append(vpd, d);
	return vpd


def euclidean_by_cycle(neuron1_trace, neuron2_trace, neuron1_beg):
    euclidean_by_cycle = []
    for i in range(len(neuron1_beg)-1):
        #DEFINE THE PART OF THE VOLTAGE TRACE THAT BELONGS TO THIS CYCLE
        PD1 = neuron1_trace[int(neuron1_beg[i]*10):int(neuron1_beg[i+1]*10)]
        PD2 = neuron2_trace[int(neuron1_beg[i]*10):int(neuron1_beg[i+1]*10)]

        #NORMALIZATIONS
        #area = np.sum(PD1)
        #area_PD1_trace = PD1/-area
        #area = np.sum(PD2)
        #area_PD2_trace = PD2/-area

        #minmax_PD1_trace = (PD1 - np.min(PD1)) / (np.max(PD1) - np.min(PD1))
        #minmax_PD2_trace = (PD2 - np.min(PD2)) / (np.max(PD2) - np.min(PD2))

        zPD1 = scp.stats.zscore(PD1)
        zPD2 = scp.stats.zscore(PD2)
        
        euclidean = np.sqrt(np.sum(np.square(zPD1-zPD2)))

        euclidean_by_cycle = np.hstack((euclidean_by_cycle,euclidean))
    return euclidean_by_cycle


def euclidean_by_burst(neuron1_trace, neuron2_trace, neuron1_beg, neuron1_end):
    euclidean_by_burst = []
    for i in range(len(neuron1_beg)):
        #DEFINE THE PART OF THE VOLTAGE TRACE THAT BELONGS TO THIS CYCLE
        PD1 = neuron1_trace[int(neuron1_beg[i]*10):int(neuron1_end[i]*10)]
        PD2 = neuron2_trace[int(neuron1_beg[i]*10):int(neuron1_end[i]*10)]

        #NORMALIZATIONS
        #area = np.sum(PD1)
        #area_PD1_trace = PD1/-area
        #area = np.sum(PD2)
        #area_PD2_trace = PD2/-area

        #minmax_PD1_trace = (PD1 - np.min(PD1)) / (np.max(PD1) - np.min(PD1))
        #minmax_PD2_trace = (PD2 - np.min(PD2)) / (np.max(PD2) - np.min(PD2))

        #Z-SCORE NORMALIZATION IS THE MOST ACCURATE

        zPD1 = scp.stats.zscore(PD1)
        zPD2 = scp.stats.zscore(PD2)
        
        euclidean = np.sqrt(np.sum(np.square(zPD1-zPD2)))

        euclidean_by_burst = np.hstack((euclidean_by_burst,euclidean))
    return euclidean_by_burst



# ----------------------------
#  HELPERS
# ----------------------------

def normalize_interval_by_mean(array):
    interval_mean = np.mean(array)
    norm_interval = np.array(array)/interval_mean
    return norm_interval


def moving_std(arr, window_size = 10):
    arr = np.array(arr)
    return np.array([np.std(arr[i:i+window_size]) for i in range(len(arr) - window_size + 1)])

def moving_coefvar(arr, window_size = 10):
    arr = np.array(arr)
    return np.array([np.std(arr[i:i+window_size])/np.mean(arr[i:i+window_size]) for i in range(len(arr) - window_size + 1)])

def moving_mean(arr, window_size = 10):
    arr = np.array(arr)
    return np.array([np.mean(arr[i:i+window_size]) for i in range(len(arr) - window_size + 1)])

def relationship_metrics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = scp.stats.linregress(x, y)
    
    slope = result.slope
    intercept = result.intercept
    r = result.rvalue
    r_squared = r ** 2
    p_value = result.pvalue
    
    y_pred = slope * x + intercept
    residuals = y - y_pred

    return slope, intercept, r, r_squared, p_value, y_pred, residuals



# ----------------------------
#  ARRAY HELPERS
# ----------------------------

def resample_array(array, target_length):
    #resample array from its original length into the given target length
    original_indices = np.linspace(0, 1, num = len(array))
    target_indices = np.linspace(0, 1, num = target_length)
    return np.interp(target_indices, original_indices, array)

def bin_array(array, num_bins = 100):
    #subdivide array into bins
    array = np.array(array)
    bins = np.array_split(array, num_bins)
    return np.array([b.mean() if len (b) > 0 else np.nan for b in bins])

def flip_metric(x):
    #mirror metric around its midpoint so that the range stays the same but order is reversed
    x = np.asarray(x, dtype=float)
    return np.nanmin(x) + np.nanmax(x) - x


def build_dictionaries(cost, interburst_threshold, PD1_spikes, PD2_spikes, LP_spikes, PD1_beg, 
                       PD1_end, PD2_beg, PD2_end, LP_beg, LP_end, PD1_trace, PD2_trace, LP_trace):
    #This function runs all the computation and anaylsis needed to create the neurons_data_dict and sync_dict.


    PD1_relation = burst_spike_relation(PD1_beg,PD1_end, PD1_spikes, offset = 80)
    PD2_relation = burst_spike_relation(PD2_beg,PD2_end, PD2_spikes, offset = 80)
    LP_relation= burst_spike_relation(LP_beg, LP_end, LP_spikes, offset = 80)

    PD1_ISIs = ISIs(PD1_relation[:,1], interburst_threshold)
    PD1_IBIs = IBIs(PD1_beg, PD1_end, interburst_threshold)
    PD2_ISIs = ISIs(PD2_relation[:,1], interburst_threshold)
    PD2_IBIs = IBIs(PD2_beg, PD2_end, interburst_threshold)
    LP_ISIs = ISIs(LP_relation[:,1], interburst_threshold)
    LP_IBIs = IBIs(LP_beg, LP_end, interburst_threshold)

    vpd = victor_purpura_distance(PD1_relation, PD2_relation)
    euclid_by_burst = euclidean_by_burst(PD1_trace, PD2_trace, PD1_beg, PD1_end)
    euclid_by_cycle = euclidean_by_cycle(PD1_trace, PD2_trace, PD1_beg)
        
    PD1_avg_ISIs = avg_ISIs(PD1_beg, PD1_relation)
    PD2_avg_ISIs = avg_ISIs(PD2_beg, PD2_relation)
    LP_avg_ISIs = avg_ISIs(LP_beg, LP_relation)

    PD1_spikes_per_burst = spike_number_per_burst(PD1_beg, PD1_relation)
    PD2_spikes_per_burst = spike_number_per_burst(PD2_beg, PD2_relation)
    LP_spikes_per_burst = spike_number_per_burst(LP_beg, LP_relation)

    PD1_sdf_1s,_ = spike_density_function(PD1_spikes, PD1_trace, PD1_beg, fs=10000, window_ms=1000, window_type="squared")
    PD2_sdf_1s,_ = spike_density_function(PD2_spikes, PD2_trace, PD2_beg, fs=10000, window_ms=1000, window_type="squared")
    LP_sdf_1s,_ = spike_density_function(LP_spikes, LP_trace, LP_beg, fs=10000, window_ms=1000, window_type="squared")

    PD1_sdf_100ms,_ = spike_density_function(PD1_spikes, PD1_trace, PD1_beg, fs=10000, window_ms=100, window_type="squared")
    PD2_sdf_100ms,_ = spike_density_function(PD2_spikes, PD2_trace, PD2_beg, fs=10000, window_ms=100, window_type="squared")
    LP_sdf_100ms,_ = spike_density_function(LP_spikes, LP_trace, LP_beg, fs=10000, window_ms=100, window_type="squared")

    neurons_data_dict = {
    "PD1": {
        "relation": PD1_relation,
        "ISIs": PD1_ISIs,
        "IBIs": PD1_IBIs,
        "avg_ISIs": PD1_avg_ISIs,
        "spikes_per_burst": PD1_spikes_per_burst,
        "spikes": PD1_spikes,
        "beg": PD1_beg,
        "end": PD1_end,
        "trace": PD1_trace,
        'sdf_1s': PD1_sdf_1s,
        'sdf_100ms': PD1_sdf_100ms
    },
    "PD2": {
        "relation": PD2_relation,
        "ISIs": PD2_ISIs,
        "IBIs": PD2_IBIs,
        "avg_ISIs": PD2_avg_ISIs,
        "spikes_per_burst": PD2_spikes_per_burst,
        "spikes": PD2_spikes,
        "beg": PD2_beg,
        "end": PD2_end,
        "trace": PD2_trace,
        'sdf_1s': PD2_sdf_1s,
        'sdf_100ms': PD2_sdf_100ms
    },
    "LP": {
        "relation": LP_relation,
        "ISIs": LP_ISIs,
        "IBIs": LP_IBIs,
        "avg_ISIs": LP_avg_ISIs,
        "spikes_per_burst": LP_spikes_per_burst,
        "spikes": LP_spikes,
        "beg": LP_beg,
        "end": LP_end,
        "trace": LP_trace,
        'sdf_1s': LP_sdf_1s,
        'sdf_100ms': LP_sdf_100ms
    }
    }

    sync_dict = {
    "vpd": vpd,
    "euclid_by_burst": euclid_by_burst,
    "euclid_by_cycle": euclid_by_cycle
    }

    return neurons_data_dict, sync_dict

def intervals_PD_reference(PD1_beg, PD1_end, PD2_beg, PD2_end, LP_beg, LP_end):
    #sanity check which one is before PD or LP (in this case it should start with PD)
    #If first PD burst is higher than first LP (PD after LP), we ignore first LP cycle
    if PD1_beg[0] > LP_beg[0]:
          LP_beg = LP_beg[1:]
          LP_end = LP_end[1:]
    
    LP_period = []
    LP_burst = []
    LP_hyperpol = []
    PD1_period = []
    PD1_burst = []
    PD1_hyperpol = []
    PD2_period = []
    PD2_burst = []
    PD2_hyperpol = []

    LPPD1_delay = []
    LPPD1_interval = []
    PD1LP_delay = []
    PD1LP_interval = []

    LPPD2_delay = []
    LPPD2_interval = []
    PD2LP_delay = []
    PD2LP_interval = []

    for i in range(len(PD1_beg) - 1):
                
        #Append value of intervals		
        LP_period.append(LP_beg[i+1] - LP_beg[i])
        LP_burst.append(LP_end[i] - LP_beg[i])
        LP_hyperpol.append(LP_beg[i+1] - LP_end[i])
        PD1_period.append(PD1_beg[i+1] - PD1_beg[i])
        PD1_burst.append(PD1_end[i] - PD1_beg[i])
        PD1_hyperpol.append(PD1_beg[i+1] - PD1_end[i])
        PD2_period.append(PD2_beg[i+1] - PD2_beg[i])
        PD2_burst.append(PD2_end[i] - PD2_beg[i])
        PD2_hyperpol.append(PD2_beg[i+1] - PD2_end[i])

        LPPD1_delay.append(PD1_beg[i+1] - LP_end[i])
        LPPD1_interval.append(PD1_beg[i+1] - LP_beg[i])
        PD1LP_delay.append(LP_beg[i] - PD1_end[i])
        PD1LP_interval.append(LP_beg[i] - PD1_beg[i])

        LPPD2_delay.append(PD2_beg[i+1] - LP_end[i])
        LPPD2_interval.append(PD2_beg[i+1] - LP_beg[i])
        PD2LP_delay.append(LP_beg[i] - PD2_end[i])
        PD2LP_interval.append(LP_beg[i]- PD2_beg[i])

    intervals_both_dict = {
        "LP_period" : LP_period,
        "LP_burst" : LP_burst,
		"LP_hyperpolarization" : LP_hyperpol,
		"PD1_period" : PD1_period,
		"PD1_burst" : PD1_burst,
		"PD1_hyperpolarization" : PD1_hyperpol,
		"PD2_period" : PD2_period,
		"PD2_burst" : PD2_burst,
		"PD2_hyperpolarization" : PD2_hyperpol,
		"LPPD1_delay" : LPPD1_delay,
		"LPPD1_interval" : LPPD1_interval,
		"PD1LP_delay" : PD1LP_delay,
		"PD1LP_interval" : PD1LP_interval,
		"LPPD2_delay" : LPPD2_delay,
		"LPPD2_interval" : LPPD2_interval,
		"PD2LP_delay" : PD2LP_delay,
		"PD2LP_interval" : PD2LP_interval}
		
    intervals_PD1_dict = {
        "LP_period" : LP_period,
        "LP_burst" : LP_burst,
        "LP_hyperpolarization" : LP_hyperpol,
        "PD1_period" : PD1_period,
        "PD1_burst" : PD1_burst,
        "PD1_hyperpolarization" : PD1_hyperpol,
        "LPPD1_delay" : LPPD1_delay,
        "LPPD1_interval" : LPPD1_interval,
        "PD1LP_delay" : PD1LP_delay,
        "PD1LP_interval" : PD1LP_interval}
            
    intervals_PD2_dict = {
        "LP_period" : LP_period,
        "LP_burst" : LP_burst,
        "LP_hyperpolarization" : LP_hyperpol,
        "PD2_period" : PD2_period,
        "PD2_burst" : PD2_burst,
        "PD2_hyperpolarization" : PD2_hyperpol,
        "LPPD2_delay" : LPPD2_delay,
        "LPPD2_interval" : LPPD2_interval,
        "PD2LP_delay" : PD2LP_delay,
        "PD2LP_interval" : PD2LP_interval}
    
    return intervals_both_dict, intervals_PD1_dict, intervals_PD2_dict



#FUNCTIONS TO EXTRACT DATA FROM NESTED DICTIONARIES
def get_nested_value(dictionary, key_path):
    #safely access nested dictionary values using a list of keys
    try:
        return reduce(operator.getitem, key_path, dictionary)
    except (KeyError, TypeError):
        return None

def extract_experiment_array(df, key_path, slicing=None):
    
    #extract a list of values/arrays from each experiment based on a nested key path, can also slice like [:-1], [1:], etc

    data_list = []
    exp_keys = []

    for exp_key, exp_data in df.items():
        value = get_nested_value(exp_data, key_path)
        if value is not None:
            try:
                if slicing:
                    # Apply slice if possible
                    value = value[slicing]
            except Exception:
                pass  # ignore if slicing fails
            data_list.append(value)
            exp_keys.append(exp_key)

    return data_list, exp_keys

def parse_slice(slice_str):
    #Convert a string like '[:-1]' or '[1:5]' into a slice object.
    slice_str = slice_str.strip("[]")
    parts = slice_str.split(":")

    def to_int_or_none(s):
        return int(s) if s else None

    if len(parts) == 1:
        return int(parts[0])  # e.g. [0]
    elif len(parts) in (2, 3):
        return slice(*[to_int_or_none(p) for p in parts])
    else:
        raise ValueError(f"Invalid slice string: {slice_str}")


#--------------SUBDIVISON FUNCTIONS-----------------

def subdivide_array_by_cycles(arr, chunk_size=50):
    #Subdivide an array into chunks of a fixed given size
    arr = np.asarray(arr)
    n = len(arr)
    
    chunks = []
    indices = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(arr[start:end])
        indices.append((start, end))
    return chunks, indices


def chunks_mean(chunks):
    return np.array([np.mean(chunk) for chunk in chunks])



def compute_metrics(
    df_data,
    metric_specs,
    chunk_size=50,
    rolling_window=None,
    rolling_step=None,
    rolling_mode="square",
    compute_cycles=True,
    compute_chunked=True,
    compute_rolling=True,
    compute_full=True,
    exps_to_ignore=None,
    fillna_cycles=None,
):
    #All in one solution to compute metrics from nested df_data across different time scales, returning all outputs in wide format: Cycles, chunked, rolling, full: one column per metric
    #Used better to compute and have all in the same df, not recommended for individual treatment of each time scale. Better use their specific functions if treated individually.
    

    if exps_to_ignore is None:
        exps_to_ignore = []

    df_wide_dict = {}

    # --- helpers ---
    def pearson_r2_vec(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        r = np.corrcoef(a[mask], b[mask])[0,1]
        return r*r

    def resolve_paths(expdata, paths, reducer):
        if reducer == "pearson_r2":  # binary/invariant
            arr1 = expdata
            arr2 = expdata
            for p in paths[0]:
                arr1 = arr1[p]
            for p in paths[1]:
                arr2 = arr2[p]
            return np.asarray(arr1), np.asarray(arr2)
        else:  # unary
            arr = expdata
            for p in paths:
                arr = arr[p]
            return np.asarray(arr)

    def get_weights(window_size, mode):
        if mode == "square":
            return np.ones(window_size)
        elif mode == "gaussian":
            x = np.arange(window_size)
            center = (window_size-1)/2
            sigma = window_size/4
            w = np.exp(-0.5*((x-center)/sigma)**2)
            return w / w.sum()
        else:
            raise ValueError(f"Unknown rolling_mode: {mode}")

    rolling_weights = get_weights(rolling_window, rolling_mode) if rolling_window else None

    # -------------------- main loop --------------------
    for exp_name, expdata in df_data.items():
        if exp_name in exps_to_ignore:
            continue

        # --- resolve arrays ---
        arrays_unary = {}
        arrays_binary = {}
        for spec in metric_specs:
            reducer = spec["reducer"]
            if reducer == "pearson_r2":
                arr1, arr2 = resolve_paths(expdata, spec["paths"], reducer)
                arrays_binary[spec["name"]] = (arr1, arr2)
            else:
                arr = resolve_paths(expdata, spec["paths"], reducer)
                arrays_unary[spec["name"]] = arr

        # -------------------- CYCLES --------------------
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

        # -------------------- CHUNKED --------------------
        if compute_chunked:
            df_chunked_dict = {"exp": [], "chunk": []}
            for name in arrays_unary.keys():
                df_chunked_dict[name] = []
            for name in arrays_binary.keys():
                df_chunked_dict[name] = []

            n_chunks = {name: len(arr)//chunk_size for name, arr in arrays_unary.items()}
            n_chunks_binary = {name: min(len(a), len(b))//chunk_size for name, (a,b) in arrays_binary.items()}
            max_chunks = max(list(n_chunks.values()) + list(n_chunks_binary.values()) + [0])

            for i in range(max_chunks):
                df_chunked_dict["exp"].append(exp_name)
                df_chunked_dict["chunk"].append(i)
                for name, arr in arrays_unary.items():
                    reducer = next(s["reducer"] for s in metric_specs if s["name"]==name)
                    if i < n_chunks[name]:
                        chunk_arr = arr[i*chunk_size:(i+1)*chunk_size]
                        df_chunked_dict[name].append(reducer(chunk_arr))
                    else:
                        df_chunked_dict[name].append(np.nan)
                for name, (a,b) in arrays_binary.items():
                    if i < n_chunks_binary[name]:
                        df_chunked_dict[name].append(pearson_r2_vec(a[i*chunk_size:(i+1)*chunk_size],
                                                                    b[i*chunk_size:(i+1)*chunk_size]))
                    else:
                        df_chunked_dict[name].append(np.nan)

            df_wide_dict.setdefault("chunked", []).append(pd.DataFrame(df_chunked_dict))

        # -------------------- ROLLING --------------------
        if compute_rolling:
            if rolling_window is None or rolling_step is None:
                raise ValueError("rolling_window and rolling_step must be set")
            df_rolling_dict = {"exp": [], "window": []}
            for name in arrays_unary.keys():
                df_rolling_dict[name] = []
            for name in arrays_binary.keys():
                df_rolling_dict[name] = []

            n_windows = {name: max((len(arr)-rolling_window)//rolling_step+1,0) for name, arr in arrays_unary.items()}
            n_windows_binary = {name: max((min(len(a),len(b))-rolling_window)//rolling_step+1,0) for name,(a,b) in arrays_binary.items()}
            max_windows = max(list(n_windows.values()) + list(n_windows_binary.values()) + [0])

            for i in range(max_windows):
                df_rolling_dict["exp"].append(exp_name)
                df_rolling_dict["window"].append(i)
                for name, arr in arrays_unary.items():
                    reducer = next(s["reducer"] for s in metric_specs if s["name"]==name)
                    if i < n_windows[name]:
                        window_arr = arr[i*rolling_step:i*rolling_step+rolling_window] * rolling_weights
                        df_rolling_dict[name].append(reducer(window_arr))
                    else:
                        df_rolling_dict[name].append(np.nan)
                for name, (a,b) in arrays_binary.items():
                    if i < n_windows_binary[name]:
                        arr1_win = a[i*rolling_step:i*rolling_step+rolling_window] * rolling_weights
                        arr2_win = b[i*rolling_step:i*rolling_step+rolling_window] * rolling_weights
                        df_rolling_dict[name].append(pearson_r2_vec(arr1_win, arr2_win))
                    else:
                        df_rolling_dict[name].append(np.nan)

            df_wide_dict.setdefault("rolling", []).append(pd.DataFrame(df_rolling_dict))

        # -------------------- FULL --------------------
        if compute_full:
            row = {"exp": exp_name, "full": -1}
            for name, arr in arrays_unary.items():
                reducer = next(s["reducer"] for s in metric_specs if s["name"]==name)
                row[name] = reducer(arr)
            for name, (a,b) in arrays_binary.items():
                row[name] = pearson_r2_vec(a,b)
            df_wide_dict.setdefault("full", []).append(pd.DataFrame([row]))

    # -------------------- concatenate per type --------------------
    for key in df_wide_dict.keys():
        df_wide_dict[key] = pd.concat(df_wide_dict[key], ignore_index=True)

    return df_wide_dict
