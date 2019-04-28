"""
records.py
-------
This module provides functions for visualizing records and labels.
By: Sebastian D. Goodfellow and Noel Kippers, 2019
"""

# 3rd party imports
import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from ipywidgets import interact, fixed
from ipywidgets.widgets import IntSlider

# Local imports
from kardionet import DATA_DIR
from kardionet.data.record import Record


def plot_record(record_name_id, record_names):
    """Plot waveform with labels."""
    # Get record name
    record_name = record_names[record_name_id]

    # Initialize record
    record = Record(record_name=record_name)

    # Setup figure
    fig = plt.figure(figsize=(15, 20), facecolor='w')
    fig.subplots_adjust(wspace=0, hspace=0.3)
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0))
    ax3 = plt.subplot2grid((4, 1), (2, 0))
    ax4 = plt.subplot2grid((4, 1), (3, 0))

    # Get time array
    time = np.arange(record.waveforms.shape[0]) * 1 / record.fs

    # Plot channel 1
    ax1.set_title('Labeled Intervals: {}'.format(record.num_intervals), fontsize=20, y=1.02, loc='left')
    ax1.plot(time, record.waveforms[:, 0], '-', color=[0.7, 0.7, 0.7], lw=2)
    for interval in record.intervals_df['interval'].unique():
        ax1.plot(time[record.intervals_df['index'][record.intervals_df['interval'] == interval]],
                 record.intervals_df['ch1'][record.intervals_df['interval'] == interval],
                 '-', color='k', lw=2)
    ax1.set_xlabel('Time, s', fontsize=22)
    ax1.set_ylabel('Ch1 Amplitude', fontsize=22)
    ax1.set_xlim([time.min(), time.max()])
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)

    # Plot channel 2
    ax2.set_title('Labeled Intervals: {}'.format(record.num_intervals), fontsize=20, y=1.02, loc='left')
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

    # Plot channel 1 labels
    time = np.arange(record.intervals_df.shape[0]) * 1 / record.fs
    ax3.set_title('Beat Labels: {}'.format(len(record.labels)), fontsize=20, y=1.02, loc='left')
    generate_line_plot(x=time, y=record.intervals_df['ch1'].values, z=record.intervals_df['train_label'].values, ax=ax3)
    ax3.set_xlabel('Time, s', fontsize=22)
    ax3.set_ylabel('Ch1 Amplitude', fontsize=22)
    ax3.set_xlim([time.min(), time.max()])
    ax3.xaxis.set_tick_params(labelsize=16)
    ax3.yaxis.set_tick_params(labelsize=16)

    # Plot channel 2 labels
    ax4.set_title('Beat Labels: {}'.format(len(record.labels)), fontsize=20, y=1.02, loc='left')
    generate_line_plot(x=time, y=record.intervals_df['ch2'].values, z=record.intervals_df['train_label'].values, ax=ax4)
    ax4.set_xlabel('Time, s', fontsize=22)
    ax4.set_ylabel('Ch2 Amplitude', fontsize=22)
    ax4.set_xlim([time.min(), time.max()])
    ax4.xaxis.set_tick_params(labelsize=16)
    ax4.yaxis.set_tick_params(labelsize=16)

    plt.show()


def generate_line_plot(x, y, z, ax):
    """Return line plot with categorical coloring."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(['k', 'r', 'b', 'g'])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    custom_lines = [Line2D([0], [0], color='k', lw=4), Line2D([0], [0], color='r', lw=4),
                    Line2D([0], [0], color='b', lw=4), Line2D([0], [0], color='g', lw=4)]
    ax.legend(custom_lines, ['NA', 'P-Wave', 'QRS-Wave', 'T-Wave'], frameon=False,
              fontsize=12, ncol=4, bbox_to_anchor=(1.013, 1.12))


def plot_records():
    """Launch interactive plotting widget."""
    # Get list of record names
    record_names = [file.split('.')[0] for file in os.listdir(os.path.join(DATA_DIR, 'raw')) if '.dat' in file]

    _ = interact(
        plot_record,
        record_name_id=IntSlider(value=0, min=0, max=len(record_names)-1, description='record_name', disabled=False),
        record_names = fixed(record_names)
    )