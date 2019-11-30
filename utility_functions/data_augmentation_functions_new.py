import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
np.set_printoptions(suppress=True)
import math
import itertools
import pickle
import h5py
from collections import namedtuple
from collections import defaultdict
import spikeextractors as se

def get_peak_events(waveforms, candidate_channels, amp_jitter=0.0, spike_jitter=8):
    min_peaks_all = np.min(waveforms[candidate_channels, waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter + 1],1)
    min_peak = np.min(min_peaks_all)
    amp_diffs = min_peaks_all - min_peak
    min_channels_ids = np.where(amp_diffs <= amp_jitter)
    min_channels = candidate_channels[min_channels_ids]
    min_peaks = min_peaks_all[min_channels_ids]
    min_frames = waveforms.shape[1]//2 - spike_jitter + np.argmin(waveforms[candidate_channels[min_channels_ids], waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter+ 1], 1)
    return min_frames, min_channels, min_peaks

def getMaxEnergyEvents(waveforms, channels, energy_jitter=0.0, energy_start_frame=10, energy_end_frame=10):
    max_energies_all = -np.sum(waveforms[channels, waveforms.shape[1]//2 - energy_start_frame:waveforms.shape[1]//2 + energy_end_frame].clip(max=0), axis=1)
    max_energy = np.max(max_energies_all)
    energy_diffs = -max_energies_all + max_energy
    max_channels_ids = np.where(energy_diffs <= energy_jitter)
    max_channels = channels[max_channels_ids]
    max_energies = max_energies_all[max_channels_ids]
    max_frames = waveforms.shape[1]//2 - energy_start_frame + np.argmin(waveforms[channels[max_channels_ids], waveforms.shape[1]//2 - energy_start_frame:waveforms.shape[1]//2 + energy_end_frame], 1)
    return max_frames, max_channels, -max_energies

def get_channel_square(chosen_channel, padded_channel_positions, width=40):
    center_position = padded_channel_positions[chosen_channel]
    
    #Define bounding box around central channel
    ll = np.asarray([center_position[1] - width, center_position[2] - width])
    ur = np.asarray([center_position[1] + width, center_position[2] + width])
    
    #Get all points within bounding box
    pts = padded_channel_positions
    inidx = np.all(np.logical_and(ll < pts[:,1:3], pts[:,1:3] < ur), axis=1)
    inbox = pts[inidx]
    channel_indices = np.where(inidx)[0]
    
    final_channels = []
    for i in range(inbox.shape[0]):
        final_channels.append((channel_indices[i], inbox[i]))
    
    return final_channels, center_position

def CreateFileDataset(args, sorted_widths):
    channel_string = str(args.width) +"um"
    train_path = args.save_directory + 'model_data_ngt_'+ channel_string + '_VAE_'+ \
                 str(args.amp_jitter)+'_amp_jitter_' + args.recording_name
    #Training data
    hf_train = h5py.File(train_path, 'w')
    #calculate number of channels in the square close to the center, according to width
    space_between_channels = (sorted_widths[-1] - sorted_widths[0]) / (len(sorted_widths) - 1)
    num_channels = (args.width // space_between_channels * 2 + 1)**2
    hf_train.create_dataset('amps_list', (args.num_spikes, num_channels, 2))
    hf_train.create_dataset('channel_locations_list', (args.num_spikes, num_channels, 3))
    hf_train.create_dataset('waveforms_list', (args.num_spikes, num_channels, args.len_snippet))
    hf_train.create_dataset('center_location_list', (args.num_spikes, 3))
    hf_train.create_dataset('central_channel_list', (args.num_spikes,),'i')
    hf_train.create_dataset('spike_time_list', (args.num_spikes,),'i')
    hf_train.create_dataset('spike_id_list', (args.num_spikes,),'i')
    return hf_train, train_path

def SavetoFile(idx, args, hf, amps_list, channel_locations_list, center_location_list, \
               central_channel_list, waveforms_list_list, peak_channel_list,spike_time_list, spike_id_list):
    idx = idx + 1
    if idx == args.num_spikes or idx % args.save_every == 0:                
        if (idx == args.num_spikes) and (idx % args.save_every):
            args.save_every = idx % args.save_every                        
        hf['waveforms_list'][idx-args.save_every:idx] = waveforms_list_list
        hf['amps_list'][idx-args.save_every:idx] = amps_list
        hf['channel_locations_list'][idx-args.save_every:idx] = channel_locations_list
        hf['center_location_list'][idx-args.save_every:idx] = center_location_list
        hf['spike_time_list'][idx-args.save_every:idx] = spike_time_list
        hf['spike_id_list'][idx-args.save_every:idx] = spike_id_list
        hf['central_channel_list'][idx-args.save_every:idx] = central_channel_list
        waveforms_list_list[:] = []
        amps_list[:] = []
        channel_locations_list[:] = []
        center_location_list[:] = []
        central_channel_list[:] = []
        peak_channel_list[:] = []
        spike_time_list[:] = []
        spike_id_list[:] = []