"""
record.py
This module provides a class and methods for processing a database record.
By: Sebastian D. Goodfellow and Noel Kippers, 2019
"""

# 3rd party imports
import os
import copy
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Local imports
from kardionet import DATA_DIR


class Record(object):

    def __init__(self, record_name, gap_tolerance=1):

        # Set parameters
        self.record_name = record_name
        self.load_path = os.path.join(DATA_DIR, 'raw')
        self.save_path = os.path.join(DATA_DIR, 'processed')
        self.gap_tolerance = gap_tolerance

        # Set attributes
        self.label_dict = {'na': 0, 'p': 1, 'N': 2, 'A': 2, 't': 3}
        self.files = self._get_files()
        self.extensions = self._get_extensions()
        self.annotator = self._get_annotator(annotator=None, auto_return=True)
        self.waveforms = self._get_waveforms()
        self.fs = self._get_sample_frequency()
        self.num_channels = self._get_channel_count()
        self.annotations = self._get_annotations()
        self.labels = self._get_labels()
        self.intervals = self._get_intervals()
        self.num_intervals = len(self.intervals)
        self.intervals_df = self._get_intervals_df()

    def save_csv(self):
        """Save intervals DataFrame to CSV."""
        self.intervals_df.to_csv(os.path.join(self.save_path, '{}.csv'.format(self.record_name)), index=False)

    def _get_files(self):
        """Get list of files associated with record_name."""
        return [file for file in os.listdir(self.load_path) if self.record_name in file]

    def _get_extensions(self):
        """Return a list file extensions for record_name."""
        return [file.split('.')[-1] for file in self.files]

    def _get_annotator(self, annotator, auto_return):
        """Return annotator file extension."""
        if annotator is not None and annotator in self.extensions:
            return annotator
        elif auto_return is True:
            if 'q1c' in self.extensions:
                return 'q1c'
            elif 'q2c' in self.extensions:
                return 'q2c'
            elif 'qt1' in self.extensions:
                return 'qt1'
            elif 'qt2' in self.extensions:
                return 'qt2'
        else:
            return None

    def _get_waveforms(self):
        """Return waveforms as numpy array."""
        return wfdb.rdrecord(os.path.join(self.load_path, self.record_name)).__dict__['p_signal']

    def _get_sample_frequency(self):
        """Return sample frequency."""
        return wfdb.rdrecord(os.path.join(self.load_path, self.record_name)).__dict__['fs']

    def _get_channel_count(self):
        """Return number of channels."""
        return wfdb.rdrecord(os.path.join(self.load_path, self.record_name)).__dict__['n_sig']

    def _get_annotations(self):
        """Return annotations."""
        if self.annotator is not None:
            return wfdb.rdann(os.path.join(self.load_path, self.record_name), self.annotator).__dict__
        else:
            return None

    def _get_labels(self):
        """Return p, N, A, and t labels."""
        if self.annotations is not None:

            # Collect labels p, N, A, and t labels
            labels = [{'peak': self.annotations['sample'][index], 'start': self.annotations['sample'][index-1],
                       'end': self.annotations['sample'][index+1], 'label': self.annotations['symbol'][index]}
                      for index in self._get_label_indices()]

            # Add gap labels
            labels = self._add_gap_labels(labels=labels)

            return labels
        else:
            return None

    def _get_label_indices(self):
        """Return the indices of p, N, A, and t labels."""
        return np.where((np.array(self.annotations['symbol']) == 'p') |
                        (np.array(self.annotations['symbol']) == 'N') |
                        (np.array(self.annotations['symbol']) == 'A') |
                        (np.array(self.annotations['symbol']) == 't'))[0]

    def _add_gap_labels(self, labels):
        """Add gap labels 'na', 'break', 'None' in between beat labels."""
        # Add None gap labels between every beat label
        labels = [x for y in (labels[i:i+1] + [None] * (i < len(labels) - 1) for i in range(len(labels))) for x in y]

        # Add 'na' and 'break' labels
        for idx in np.where(np.array(labels) == None)[0]:

            if (labels[idx + 1]['start'] - labels[idx - 1]['end']) * 1 / self.fs > self.gap_tolerance:
                labels[idx] = 'break'

            elif (self._check_beat_order(previous_beat=labels[idx-1]['label'], next_beat=labels[idx+1]['label']) and
                  labels[idx + 1]['start'] != labels[idx - 1]['end']):
                labels[idx] = {'peak': None, 'start': labels[idx-1]['end'],
                               'end': labels[idx+1]['start'], 'label': 'na'}
            else:
                pass

        # Remove None gap for contiguous labels
        labels = [label for label in labels if label is not None]

        return labels

    @staticmethod
    def _check_beat_order(previous_beat, next_beat):
        """Check that the beat order around an interval is correct."""
        if previous_beat == 'p' and next_beat == 'N':
            return True
        elif previous_beat == 't' and next_beat == 'A':
            return True
        elif previous_beat == 'N' and next_beat == 't':
            return True
        elif previous_beat == 'A' and next_beat == 't':
            return True
        elif previous_beat == 't' and next_beat == 'p':
            return True
        else:
            return False

    def _get_intervals(self):
        """Return contiguous intervals of labels as lists."""
        if any('break' in label for label in self.labels):
            labels = copy.copy(self.labels)
            labels.append('')
            sublists = list()
            intervals = list()
            for label in labels:
                if label != 'break':
                    sublists.append(label)
                else:
                    intervals.append(sublists)
                    sublists = list()
            return intervals
        else:
            return [copy.copy(self.labels)]

    def _get_intervals_df(self):
        """Return contiguous intervals of labels as DataFrame."""
        # Interval lists
        index = list()
        time = list()
        waveforms = list()
        labels = list()
        train_labels = list()
        intervals = list()

        for idx, interval in enumerate(self.intervals):
            for label in interval:

                # Collect labels
                time.append(np.arange(label['end'] - label['start'], dtype=np.float64) * 1 / self.fs)
                index.append(np.arange(label['start'], label['end'], dtype=np.int64))
                waveforms.append(self.waveforms[label['start']:label['end'], :])
                labels.extend([label['label'] for _ in range(label['end'] - label['start'])])
                train_labels.extend([self.label_dict[label['label']] for _ in range(label['end'] - label['start'])])
                intervals.extend([idx for _ in range(label['end'] - label['start'])])

        # Concatenate arrays
        time = np.concatenate(time, axis=0)
        index = np.concatenate(index, axis=0)
        waveforms = np.concatenate(waveforms, axis=0)

        # Create intervals DataFrame
        return pd.DataFrame({'time': time, 'index': index, 'ch1': waveforms[:, 0], 'ch2': waveforms[:, 1],
                             'label': labels, 'train_label': train_labels, 'interval': intervals})

    def plot_waveform(self):
        """Plot waveform with labels."""
        # Setup figure
        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))

        # Get time array
        time = np.arange(self.waveforms.shape[0]) * 1 / self.fs

        # Plot channel 1
        ax1.set_title('Intervals: {}\nLabels{}')
        ax1.plot(time, self.waveforms[:, 0], '-', color=[0.7, 0.7, 0.7], lw=2)
        for interval in self.intervals_df['interval'].unique():
            ax1.plot(time[self.intervals_df['index'][self.intervals_df['interval'] == interval]],
                     self.intervals_df['ch1'][self.intervals_df['interval'] == interval],
                     '-', color='k', lw=2)

        ax1.set_ylabel('Ch1 Amplitude', fontsize=22)
        ax1.set_xlim([time.min(), time.max()])
        ax1.tick_params(labelbottom='off')
        ax1.yaxis.set_tick_params(labelsize=16)

        # Plot channel 2
        ax2.plot(time, self.waveforms[:, 1], '-', color=[0.7, 0.7, 0.7], lw=2)
        for interval in self.intervals_df['interval'].unique():
            ax2.plot(time[self.intervals_df['index'][self.intervals_df['interval'] == interval]],
                     self.intervals_df['ch2'][self.intervals_df['interval'] == interval],
                     '-', color='k', lw=2)
        ax2.set_ylabel('Ch2 Amplitude', fontsize=22)
        ax2.set_xlim([time.min(), time.max()])
        ax2.tick_params(labelbottom='off')
        ax2.yaxis.set_tick_params(labelsize=16)

        plt.show()
