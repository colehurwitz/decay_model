import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from torch.utils import data
from collections import defaultdict
from sklearn import decomposition


class EventDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, train_spikes):
        'Initialization'
        self.train_spikes = train_spikes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_spikes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        train_spike = self.train_spikes[index]
        
        amps = train_spike.amps
        waveforms = train_spike.waveforms
        ch_locs = train_spike.ch_locs
        center_loc = train_spike.center_loc
        spike_id = train_spike.spike_id
        exp_id = train_spike.exp_id
        min_amp = train_spike.min_amp
        min_waveform = train_spike.min_waveform
        
        return amps, waveforms, ch_locs, center_loc, spike_id, exp_id, min_amp, min_waveform
        
def getEstimatedLocationsGroundTruth(model, dup_spike_ids, neuron_loc_array, neuron_array, spike_time_list, overlap_array, device, overlap, testing_set, amp_threshold=0):
    #Calculate avg-dist from neuron (2D)
    model.eval()
    vae_locs = defaultdict(list)
    vae_loc_errors = defaultdict(list)
    vae_variances = defaultdict(list)
    neuron_locs = defaultdict(list)
    center_locs = defaultdict(list)
    all_waveforms_dict = defaultdict(list)
    all_spike_times = defaultdict(list)
    for i in range(len(dup_spike_ids)):
        spike_ids = dup_spike_ids[i]
        num_spikes = len(spike_ids)
        keep_spike = False
        loc_ests = np.zeros(3)
        var_ests = np.zeros(3)
        for idx in spike_ids:
            if(overlap_array[idx] == overlap or overlap == 'all'):
                if(testing_set[idx][6] <= amp_threshold):
                    keep_spike = True
                    t_amps = testing_set[idx][0].to(device).view(1, testing_set[idx][0].shape[0], testing_set[idx][0].shape[1])
                    t_ch_locs = testing_set[idx][2].to(device).view(1, testing_set[idx][2].shape[0], testing_set[idx][2].shape[1])
                    t_waveforms = testing_set[idx][1].to(device).view(1, testing_set[idx][1].shape[0], testing_set[idx][1].shape[1])
                    exp_ids = testing_set[idx][5].to(device)
                    t_center_loc = testing_set[idx][3]
                    t_min_waveforms = testing_set[idx][7].to(device).view(1, testing_set[idx][7].shape[0])
                    recon_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var = model(t_amps, t_waveforms, t_ch_locs, exp_ids)
                    loc_ests[0] += x_mu.item() + t_center_loc[0]
                    loc_ests[1] += y_mu.item() + t_center_loc[1]
                    loc_ests[2] += z_mu.item() + t_center_loc[2]
                    var_ests[0] += x_var.item()   
                    var_ests[1] += y_var.item()  
                    var_ests[2] += z_var.item()
        if(keep_spike):
            loc_ests = loc_ests/num_spikes
            var_ests = var_ests/num_spikes
            vae_locs[neuron_array[spike_ids[0]]].append([loc_ests[0], loc_ests[1], loc_ests[2]])
            vae_variances[neuron_array[spike_ids[0]]].append([var_ests[0], var_ests[1], var_ests[2]])
            all_waveforms_dict[neuron_array[spike_ids[0]]].append(testing_set[spike_ids[0]][7])
            all_spike_times[neuron_array[spike_ids[0]]].append(spike_time_list[spike_ids[0]])
            
    return vae_locs, vae_variances, all_waveforms_dict, all_spike_times

# def getEstimatedLocationsGroundTruth2(model, dup_spike_ids, neuron_loc_array, neuron_array, spike_time_list, overlap_array, device, overlap, testing_set, amp_threshold=0):
#     #Calculate avg-dist from neuron (2D)
#     model.eval()
#     vae_locs = defaultdict(list)
#     vae_loc_errors = defaultdict(list)
#     vae_variances = defaultdict(list)
#     neuron_locs = defaultdict(list)
#     center_locs = defaultdict(list)
#     all_waveforms_dict = defaultdict(list)
#     all_spike_times = defaultdict(list)
#     for i in range(len(dup_spike_ids)):
#         spike_ids = dup_spike_ids[i]
#         num_spikes = len(spike_ids)
#         keep_spike = False
#         loc_ests = np.zeros(3)
#         var_ests = np.zeros(3)
#         for idx in spike_ids:
#             if(overlap_array[idx] == overlap or overlap == 'all'):
#                 if(testing_set[idx][6] <= amp_threshold):
#                     keep_spike = True
#                     t_amps = testing_set[idx][0].to(device).view(1, testing_set[idx][0].shape[0], testing_set[idx][0].shape[1])
#                     t_ch_locs = testing_set[idx][2].to(device).view(1, testing_set[idx][2].shape[0], testing_set[idx][2].shape[1])
#                     t_waveforms = testing_set[idx][1].to(device).view(1, testing_set[idx][1].shape[0], testing_set[idx][1].shape[1])
#                     exp_ids = testing_set[idx][5].to(device)
#                     t_center_loc = testing_set[idx][3]
#                     t_min_waveforms = testing_set[idx][7].to(device).view(1, testing_set[idx][7].shape[0])
#                     recon_amps, recon_waveforms, x_mu, x_var, y_mu, y_var, z_mu, z_var = model(t_amps, t_waveforms, t_min_waveforms, t_ch_locs, exp_ids)
#                     loc_ests[0] += x_mu.item() + t_center_loc[0]
#                     loc_ests[1] += y_mu.item() + t_center_loc[1]
#                     loc_ests[2] += z_mu.item() + t_center_loc[2]
#                     var_ests[0] += x_var.item()   
#                     var_ests[1] += y_var.item()  
#                     var_ests[2] += z_var.item()
#         if(keep_spike):
#             loc_ests = loc_ests/num_spikes
#             var_ests = var_ests/num_spikes
#             vae_locs[neuron_array[spike_ids[0]]].append([loc_ests[0], loc_ests[1], loc_ests[2]])
#             vae_variances[neuron_array[spike_ids[0]]].append([var_ests[0], var_ests[1], var_ests[2]])
#             all_waveforms_dict[neuron_array[spike_ids[0]]].append(testing_set[spike_ids[0]][7])
#             all_spike_times[neuron_array[spike_ids[0]]].append(spike_time_list[spike_ids[0]])
            
#     return vae_locs, vae_variances, all_waveforms_dict, all_spike_times