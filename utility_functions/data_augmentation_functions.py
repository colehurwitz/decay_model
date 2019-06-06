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


def getPeakEvent(waveforms, channels, spike_jitter=8):
    min_channel_idx = np.argmin(np.min(waveforms[channels, waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter + 1],1))
    min_peak = np.min(waveforms[channels[min_channel_idx], waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter + 1])
    min_frame = waveforms.shape[1]//2 - spike_jitter + np.argmin(waveforms[channels[min_channel_idx], waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter+ 1])
    min_channel = channels[min_channel_idx]
    return min_frame, min_channel, min_peak


def getPeakEvents(waveforms, channels, amp_jitter=0.0, spike_jitter=8):
    min_peaks_all = np.min(waveforms[channels, waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter + 1],1)
    min_peak = np.min(min_peaks_all)
    amp_diffs = min_peaks_all - min_peak
    min_channels_ids = np.where(amp_diffs <= amp_jitter)
    min_channels = channels[min_channels_ids]
    min_peaks = min_peaks_all[min_channels_ids]
    min_frames = waveforms.shape[1]//2 - spike_jitter + np.argmin(waveforms[channels[min_channels_ids], waveforms.shape[1]//2 - spike_jitter:waveforms.shape[1]//2 + spike_jitter+ 1], 1)
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

def getChannelSquare(chosen_channel, padded_channel_positions, width=40):
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