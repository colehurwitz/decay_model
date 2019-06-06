import math 
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Computing centre of mass as prior
def getCentroid(curr_amps, curr_ch_locs, com_channels = 25):
    sum_amplitude = 0
    sum_weighted_y = 0
    sum_weighted_z = 0

    for i in range(com_channels):
        amp = abs(curr_amps[i])
        y_loc = curr_ch_locs[i][1]
        z_loc = curr_ch_locs[i][2]

        sum_amplitude += (amp**2)
        sum_weighted_y += (amp**2)*y_loc
        sum_weighted_z += (amp**2)*z_loc
        
    com_y = sum_weighted_y / sum_amplitude
    com_z = sum_weighted_z / sum_amplitude
    
    return com_y, com_z

# Computing centre of mass as prior
def getCOM(curr_amps, curr_ch_locs, com_channels = 15):
    sum_amplitude = 0
    sum_weighted_y = 0
    sum_weighted_z = 0
    
    for i in range(com_channels):
        amp = abs(curr_amps[i])
        y_loc = curr_ch_locs[i][1]
        z_loc = curr_ch_locs[i][2]

        sum_amplitude += amp
        sum_weighted_y += amp*y_loc
        sum_weighted_z += amp*z_loc
    
        com_y = sum_weighted_y / sum_amplitude
        com_z = sum_weighted_z / sum_amplitude
    
    return com_y, com_z

def getCOMEstimatedLocationsGroundTruth(testing_set, neuron_loc_array, neuron_array, overlap_array, spike_time_list, overlap='all', amp_threshold=0, com_channels=15):
    #Calculate avg-dist from neuron (2D)
    com_locs = defaultdict(list)
    com_locs_errors = defaultdict(list)
    neuron_locs = defaultdict(list)
    center_locs = defaultdict(list)
    waveforms = defaultdict(list)
    spike_times = defaultdict(list)
    for idx in range(len(testing_set)):
        if(overlap_array[idx] == overlap or overlap == 'all'):
            if(testing_set[idx][6] <= amp_threshold):
                real_amps = testing_set[idx][0][:,0][testing_set[idx][0][:,1] == 1]
                real_channels = testing_set[idx][2][testing_set[idx][0][:,1] == 1]
                real_waveforms = testing_set[idx][1][testing_set[idx][0][:,1] == 1]
                sorted_channels = real_channels[[i for i, _ in sorted(enumerate(real_channels), key=lambda x:np.linalg.norm(x[1] - [0,0,0]))]]
                sorted_amps = real_amps[[i for i, _ in sorted(enumerate(real_channels), key=lambda x:np.linalg.norm(x[1] - [0,0,0]))]]
                sorted_waveforms = real_waveforms[[i for i, _ in sorted(enumerate(real_channels), key=lambda x:np.linalg.norm(x[1] - [0,0,0]))]]
                t_center_loc = testing_set[idx][3]
                y_est, z_est = getCOM(sorted_amps, sorted_channels, com_channels)
                error = math.sqrt((y_est - neuron_loc_array[idx][1])**2 + (z_est - neuron_loc_array[idx][2])**2)
                com_locs[neuron_array[idx]].append([y_est, z_est])
                neuron_loc = neuron_loc_array[idx]
                neuron_locs[neuron_array[idx]].append([neuron_loc[0], neuron_loc[1], neuron_loc[2]])
                com_locs_errors[neuron_array[idx]].append(error)
                center_locs[neuron_array[idx]].append(t_center_loc)
                waveforms[neuron_array[idx]].append(sorted_waveforms[:com_channels])
                spike_times[neuron_array[idx]].append(spike_time_list[idx])
            
    return com_locs, com_locs_errors, center_locs, neuron_locs, waveforms, spike_times