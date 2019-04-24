"""
records.py
-------
This module provides functions for visualizing records and labels.
By: Sebastian D. Goodfellow and Noel Kippers, 2019
"""

# 3rd party imports
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from ipywidgets.widgets import IntSlider

# Local imports
from kardionet.config.config import DATA_DIR
from kardionet.data.record import Record


def plot_record(record_name_id, record_names):
    """Plot waveform with labels."""
    # Get record name
    record_name = record_names[record_name_id]

    # Initialize record
    record = Record(record_name=record_name)

    # Setup figure
    fig = plt.figure(figsize=(15, 15))
    fig.subplots_adjust(wspace=0, hspace=0.3)
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax3 = plt.subplot2grid((3, 1), (2, 0))

    # Get time array
    time = np.arange(record.waveforms.shape[0]) * 1 / record.fs

    # Plot channel 1
    ax1.set_title('Intervals: {}\nLabels: {}'.format(record.num_intervals, len(record.labels)),
                  fontsize=20, y=1.02, loc='left')
    ax1.plot(time, record.waveforms[:, 0], '-', color=[0.7, 0.7, 0.7], lw=2)
    for interval in record.intervals_df['interval'].unique():
        ax1.plot(time[record.intervals_df['index'][record.intervals_df['interval'] == interval]],
                 record.intervals_df['ch1'][record.intervals_df['interval'] == interval],
                 '-', color='k', lw=2)
    ax1.set_xlabel('Time, s', fontsize=22)
    ax1.set_ylabel('Ch1 Amplitude', fontsize=22)
    ax1.set_xlim([time.min(), time.max()])
    #ax1.axes.get_xaxis().set_visible(False)
    #ax1.tick_params(labelbottom='off')
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)

    # Plot channel 2
    ax2.plot(time, record.waveforms[:, 1], '-', color=[0.7, 0.7, 0.7], lw=2)
    for interval in record.intervals_df['interval'].unique():
        ax2.plot(time[record.intervals_df['index'][record.intervals_df['interval'] == interval]],
                 record.intervals_df['ch2'][record.intervals_df['interval'] == interval],
                 '-', color='k', lw=2)
    ax2.set_xlabel('Time, s', fontsize=22)
    ax2.set_ylabel('Ch2 Amplitude', fontsize=22)
    ax2.set_xlim([time.min(), time.max()])
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)

    # Plot labels
    ax3.set_title('Intervals', fontsize=20, y=1.02, loc='left')
    ax3.plot(record.intervals_df['ch1'], '-k')
    ax3.set_xlabel('Samples', fontsize=22)
    ax3.set_ylabel('Ch2 Amplitude', fontsize=22)
    ax3.set_xlim([0, record.intervals_df['ch1'].shape[0]])
    ax3.xaxis.set_tick_params(labelsize=16)
    ax3.yaxis.set_tick_params(labelsize=16)

    plt.show()



def plot_records():
    """Launch interactive plotting widget."""
    # Get list of record names
    record_names = [file.split('.')[0] for file in os.listdir(os.path.join(DATA_DIR, 'raw')) if '.dat' in file]

    _ = interact(
        plot_record,
        record_name_id=IntSlider(value=0, min=0, max=len(record_names)-1, description='record_name', disabled=False),
        record_names = fixed(record_names)
    )