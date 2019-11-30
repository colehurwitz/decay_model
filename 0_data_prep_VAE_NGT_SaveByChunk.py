#!/usr/bin/env python
# coding: utf-8

# # Preparing a Non-ground Truth Dataset for Localization
# 
# This notebook shows how the ground truth recordings are prepared for the paper: *Scalable Spike Source Localization in Extracellular Recordings using Amortized Variational Inference*.
# 
# For the data preparation, we perform the data augmentation described in the original manuscript. This data augmentation introduces "virtual" channels which exist outside of the MEA, in addition to the real, recording channels. We extract all "detected" events from a patch of channels near the soma of the firing neuron and then center the extracted data on the channel with the largest detected spike. This provides a realistic dataset for evaluating the performance our localization method.
# 
# We designed this notebook to be compatible with any dataset compatible with SpikeInterface.

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
np.set_printoptions(suppress=True)
import h5py
from collections import defaultdict
import ast
import spikeextractors as se
import spiketoolkit as st
import h5py
from utility_functions import data_augmentation_functions_new as fn
import argparse
import matplotlib as mpl
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 
mpl.rcParams.update({'font.size': 12})

import urllib
from urllib.request import urlretrieve


# Here you can download an example MEArec dataset that is 4.2gb. This example dataset is used throughout these notebooks

# In[2]:


parser = argparse.ArgumentParser(description='Data Prep')
parser.add_argument('--num_spikes', type=int, default=500,
                    help='number of spikes to use (default: 500. Set to -1 to use all spikes)')
parser.add_argument('--save', type=int, default=True,
                    help='Whether to save prepared data (default: True)')
parser.add_argument('--save_every', type=int, default=None, metavar='M',
                    help='save to file every M spikes (default: None - Save once at the end)')
parser.add_argument('--len_snippet', type=int, default=61,
                    help='length of a waveform snippet (default: 61)')
parser.add_argument('--width', type=int, default=40,
                    help='The distance from the channel with the largest amplitude spike \
                    for which channels are included in the constructed data (microns). (default: 40)')
parser.add_argument('--recording_directory', type=str, default='./recordings/',
                    help='path to recording directory (default: ./recordings/)')
parser.add_argument('--save_directory', type=str, default='./recordings/',
                    help='path to save directory (default: ./recordings/)')
parser.add_argument('--recording_name', type=str,                     default='recordings_300_SqMEA-10-15um_minamp0_60s_10uV_far-neurons_bpf_25-03-2019.h5',
                    help='recording file \
                    (default: recordings_300_SqMEA-10-15um_minamp0_60s_10uV_far-neurons_bpf_25-03-2019.h5)')
parser.add_argument('--spike_jitter', type=int, default=5,
                    help='The number of frames used to align the extracted waveforms. (default: 5)')
parser.add_argument('--amp_jitter', type=int, default=0,
                    help='The amplitude jitter hyperparameter. (default: 0)')
args = parser.parse_args([])


# In[ ]:


# #Example MEArec dataset to be downloaded (4.2gb)
# file_url = 'https://www.dropbox.com/s/1jolgsw5kgxmsd5/recordings_300_SqMEA-10-15um_minamp0_60s_10uV_far-neurons_bpf_25-03-2019.h5?dl=1'
# file_name = '/disk/scratch/cole/recordingsrecordings_300_SqMEA-10-15um_minamp0_60s_10uV_far-neurons_bpf_25-03-2019.h5'

# urllib.request.urlretrieve(file_url, file_name)


# Here, the path to the MEArec recording and recording name are provided

# In[3]:


recording = se.MEArecRecordingExtractor(args.recording_directory + args.recording_name, locs_2d=False)
sorting = se.MEArecSortingExtractor(args.recording_directory + args.recording_name)
channel_positions = np.asarray(recording.get_channel_locations())


# In[4]:


spike_times = []
for unit_id in sorting.get_unit_ids():
    spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_times.extend(spike_train)
spike_times = sorted(spike_times)

if args.num_spikes == -1:
    args.num_spikes = len(spike_times)
if args.save_every == None:
    args.save_every = args.num_spikes
    
#For testing
spike_times = spike_times[:args.num_spikes]


# In[5]:


#Generate min channels if not given by user (we assume it is given!)
min_channels = []
for i, spike_time in enumerate(spike_times):
    if i % (int(len(spike_times)/5)) == 0:
        print(float(i)/len(spike_times), '%')
    snippets = np.squeeze(recording.get_snippets(channel_ids=None, reference_frames=[spike_time], snippet_len=10),0)
    min_channel_id = np.argmin(np.min(snippets, 1))
    min_channels.append(min_channel_id)


# We now pad the electrode positions with "virtual" channels that lie outside the bounds of the MEA

# In[6]:


sorted_widths = np.unique(np.sort(channel_positions[:,1]))
buffer_height = sorted_widths[-1] + (-sorted_widths[0]) + (sorted_widths[-1] - sorted_widths[-2])
sorted_heights = np.unique(np.sort(channel_positions[:,2]))
buffer_width = sorted_heights[-1] + (-sorted_heights[0]) + (sorted_heights[-1] - sorted_heights[-2])

padded_channel_list = list(channel_positions)
for i in range(-1, 2):
    for j in range(-1, 2):
        if((i,j) != (0,0)):
            buffer_channel_y = channel_positions[:,1] + buffer_height*i
            buffer_channel_z = channel_positions[:,2] + buffer_width*j
            channel_positions_copy = np.copy(channel_positions)
            channel_positions_copy[:,1] += buffer_height*i
            channel_positions_copy[:,2] += buffer_width*j
            padded_channel_list = padded_channel_list + list(channel_positions_copy)

padded_channel_positions = np.asarray(padded_channel_list)

plt.figure(figsize=(12, 12))
plt.scatter(padded_channel_positions[:,1], padded_channel_positions[:,2], color='orange', marker='s', alpha=.5, label='Virtual Channel')
plt.scatter(channel_positions[:,1], channel_positions[:,2], color='blue', marker='s', label='Real Channel')
plt.legend(fancybox=True, framealpha=1);
plt.show()

# Now, we extract all the waveforms for each event in the recording and compute the closest channels to each neuron for later use

# In[7]:


width_dist_channels = defaultdict(list)
channel_ids = recording.get_channel_ids()
for i, channel in enumerate(channel_ids):
    channel_ids_copy = np.copy(channel_ids)
    closest_channels = np.asarray(sorted(channel_ids, key=lambda channel_id: np.linalg.norm(channel_positions[channel_id] - channel_positions[channel])))
    for close_channel in closest_channels:
        if np.linalg.norm(channel_positions[close_channel] - channel_positions[channel]) < args.width:
            width_dist_channels[channel].append(close_channel)


# Finally, we construct and save the augmented data in a format that can be used for our localization method. We also store the ground truth for later evaluation.

# In[8]:


if(args.save):   
    hf_train, train_path = fn.CreateFileDataset(args, sorted_widths)

amps_list = []
channel_locations_list = []
center_location_list = []
central_channel_list = []
waveforms_list_list = []
peak_channel_list = []
spike_time_list = []
spike_id_list = []
spike_id = 0
import time
t1 = time.time()
for i, min_channel in enumerate(min_channels):
    if i % (int(len(min_channels)/5)) == 0:
        print(float(i)/len(min_channels), '%')
    dists = []
    amps = []
    locations = []    
    spike_time = spike_times[i]
    candidate_channels = np.asarray(width_dist_channels[min_channel])
    waveforms = np.squeeze(recording.get_snippets(reference_frames=[spike_time], snippet_len=args.len_snippet), 0)
    peak_frames, peak_channels, peak_amps = fn.get_peak_events(waveforms, candidate_channels, amp_jitter=args.amp_jitter,spike_jitter=args.spike_jitter)
    for j, peak_channel in enumerate(peak_channels):
        peak_frame = peak_frames[j]
        peak_amp = peak_amps[j]
        #Get a group of channels within the given width from the max channel
        square_channel_tuples, center_position = fn.get_channel_square(peak_channel, padded_channel_positions, width=args.width)

        #Construct augmented data dataset with virtual and real channels (take min of each real channel (jitter around peak))
        amps = []
        channel_locations = []
        waveforms_list = []
        #If all waveforms are positive, we discard the event (this happens incredibly rarely in ground truth data and almost never in real data).
        if(peak_amp < 0):
            for sct in square_channel_tuples:
                channel = sct[0]
                scaled_position = padded_channel_positions[channel] - center_position
                channel_x = scaled_position[0]
                channel_y = scaled_position[1]
                channel_z = scaled_position[2]
                if(channel < channel_positions.shape[0]):
                    #Real channel
                    observed = 1
                    #Get min peak within the spike jitter around the frame where the true minimum occurred
                    min_peak = np.min(waveforms[channel, peak_frame - args.spike_jitter:peak_frame + args.spike_jitter]) 
                    peak_reading = min_peak
                    waveforms_list.append(waveforms[channel,:])
                else:
                    #Virtual channel
                    observed = 0
                    #Virtual reading
                    peak_reading = 0
                    waveforms_list.append(np.zeros(len(waveforms[0,:])))
                #Calculate relative location compared to center position for channel
                channel_locations.append([channel_x, channel_y, channel_z])
                #Construct augmented amplitudes for channel
                amps.append((peak_reading, observed))
            #Model data
            waveforms_list_list.append(waveforms_list)
            amps_list.append(amps)
            channel_locations_list.append(channel_locations)
            center_location_list.append(list(center_position))
            peak_channel_list.append(peak_channel)
            spike_time_list.append(spike_time)
            spike_id_list.append(spike_id)
            central_channel_list.append(peak_channel)
            if(args.save):
                fn.SavetoFile(i, args, hf_train, amps_list, channel_locations_list, center_location_list,                   central_channel_list, waveforms_list_list, peak_channel_list, spike_time_list, spike_id_list)
        else:
            pass
    spike_id += 1
# ################################################################################## Save augmented data
if(args.save):
    hf_train.close()
#     print("Train Path: " + train_path)
print(time.time()-t1)


# Now, we can visualize examples for the augmented dataset to see how the data was prepared.

# In[9]:


#load the data for plotting
hf = h5py.File(train_path, 'r')
amps_array = np.asarray(hf['amps_list'])
channel_locations_array = np.asarray(hf['channel_locations_list'])
waveforms_array = np.asarray(hf['waveforms_list'])
center_location_array = np.asarray(hf['center_location_list'])
hf.close()

scalar = .05
mid_frame = waveforms_array.shape[2]//2 # Middle frame of extracted waveform
cutout_start = 19 # Number of frames before peak to plot
cutout_end = 40 # Number of frames after peak to plot

num_plots = 6
col = 3
row = num_plots//col
if num_plots % col:
    row = row + 1
fig, ax_array = plt.subplots(row, col, figsize=(6*col, 6*row))
[ax.set_axis_off() for ax in ax_array.ravel()]

for event_id, amps in enumerate(amps_array[0:num_plots]):
    amps = amps_array[event_id]
    waveforms = waveforms_array[event_id][:,mid_frame-cutout_start:mid_frame+cutout_end]
    channel_locs = channel_locations_array[event_id]
    center_loc = center_location_array[event_id]
    ax = ax_array.ravel()[event_id]

    for i in range(len(amps)):
        ax.scatter(channel_locs[i][1] + center_loc[1], channel_locs[i][2] + center_loc[2], s=30, c="grey", marker='s')
        readings = waveforms[i]
        xs = np.linspace(-2.5, 5, waveforms[i].shape[0])
        ax.plot(channel_locs[i][1] +  center_loc[1] + xs, channel_locs[i][2] + center_loc[2] + readings*scalar, color='blue')
        observed = amps[i][1]
        text = ax.annotate(int(amps[i][1]), (channel_locs[i][1] + center_loc[1] + 1, channel_locs[i][2] + center_loc[2] + 1))
        text.set_fontsize(10)

from matplotlib.lines import Line2D
line2 = Line2D(range(1), range(1), color="white", marker='s', markerfacecolor="grey", markersize=10)
line3 = Line2D(range(1), range(1), color="blue", markerfacecolor="blue", markersize=12)
ax.legend((line2,line3),('Electrode', 'Waveform'),numpoints=1, loc=4, framealpha=1);

plt.show()







