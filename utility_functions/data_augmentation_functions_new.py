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
