"""
qtdb.py
-------
This module provides classes and methods for creating the QT database.
By: Sebastian D. Goodfellow and Noel Kippers, 2019
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import copy
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Local imports
from kardionet.config.config import DATA_DIR


class QTDB(object):

    """
    The QT Database
    https://physionet.org/physiobank/database/qtdb/
    """

    def __init__(self):

        # Set attributes
        self.db_name = 'qtdb'
        self.raw_path = os.path.join(DATA_DIR, 'raw')
        self.processed_path = os.path.join(DATA_DIR, 'processed')
        self.record_names = None

        # Create folders
        self._create_folders()

    def generate_db(self):
        """Generate raw and processed databases."""
        # Generate raw database
        # self.generate_raw_db()

        # Generate processed database
        self.generate_processed_db()

    def generate_raw_db(self):
        """Generate the raw version of the QT database in the 'raw' folder."""
        print('Generating Raw QT Database...\nSave path: {}'.format(self.raw_path))
        # Download database
        wfdb.dl_database(self.db_name, self.raw_path)

        # Get list of record names
        self.record_names = self._get_record_names()
        print('Complete!\n')

    def generate_processed_db(self):
        """Generate the processed version of the QT database in the 'processed' folder."""
        # Get record names
        self.record_names = self._get_record_names()

        # Loop through records
        for record_name in self.record_names[4:5]:

            # Process Record
            record = Record(record_name=record_name, load_path=self.raw_path, save_path=self.processed_path)
            print('')

    def _create_folders(self):
        """Create a folders for raw and processed data."""
        # Raw data path
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)
        with open(os.path.join(self.raw_path, '.gitignore'), 'w') as file:
            file.write('*\n!.gitignore')

        # Processed data path
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)
        with open(os.path.join(self.processed_path, '.gitignore'), 'w') as file:
            file.write('*\n!.gitignore')

    def _get_record_names(self):
        """Return list of records in 'raw' path."""
        return [file.split('.')[0] for file in os.listdir(self.raw_path) if '.dat' in file]


class Record(object):

    def __init__(self, record_name, load_path, save_path):

        # Set parameters
        self.record_name = record_name
        self.load_path = load_path
        self.save_path = save_path

        # Set attributes
        self.label_dict = {'na': 0, 'p': 1, 'qrs': 2, 't': 3}
        self.files = self._get_files()
        self.extensions = self._get_extensions()
        self.annotator = self._get_annotator(annotator=None, auto_return=True)
        self.waveforms = self._get_waveforms()
        self.fs = self._get_sample_frequency()
        self.num_channels = self._get_channel_count()
        self.annotations = self._get_annotations()
        self.indices = self._get_indices()
        self.p_wave_indices = self._get_p_wave_indices()
        self.qrs_wave_indices = self._get_qrs_wave_indices()
        self.t_wave_indices = self._get_t_wave_indices()
        self.p_wave_intervals = self._get_intervals(indices=self.p_wave_indices)
        self.qrs_wave_intervals = self._get_intervals(indices=self.qrs_wave_indices)
        self.t_wave_intervals = self._get_intervals(indices=self.t_wave_indices)
        self.df = self._get_df()

        print(self.indices)
        # self.plot_waveform()

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

    def _get_indices(self):
        """Return p, N, A, and t indices."""
        if self.annotations is not None:

            # Collect indices
            indices = np.where((np.array(self.annotations['symbol']) == 'p') |
                               (np.array(self.annotations['symbol']) == 'N') |
                               (np.array(self.annotations['symbol']) == 'A') |
                               (np.array(self.annotations['symbol']) == 't'))[0]

            indices = [{'index': self.annotations['sample'][index],
                        'start': self.annotations['sample'][index-1],
                        'end': self.annotations['sample'][index+1],
                        'label': self.annotations['symbol'][index]} for index in indices]

            intervals = copy.copy(indices)

            for idx, val in enumerate(indices):

                # P-wave
                if val['label'] == 'p':

                    # First annotation
                    if idx == 0:
                        if indices[idx+1]['label'] == 'N':
                            # intervals.insert(3, 'o')
                            pass

            # return pd.DataFrame(
            #     data=[(self.annotations['sample'][index], self.annotations['symbol'][index],
            #            self.annotations['sample'][index-1], self.annotations['sample'][index+1])
            #           for index in indices],
            #     columns=['index', 'label', 'start', 'end']
            # )
            return indices
        else:
            return None

    def _get_p_wave_indices(self):
        """Return p-wave indices."""
        if self.annotations is not None:
            return np.where(np.array(self.annotations['symbol']) == 'p')[0]
        else:
            return None

    def _get_qrs_wave_indices(self):
        """Return qrs-wave indices."""
        if self.annotations is not None:
            return np.where((np.array(self.annotations['symbol']) == 'N') |
                            (np.array(self.annotations['symbol']) == 'A'))[0]
        else:
            return None

    def _get_t_wave_indices(self):
        """Return t-wave indices."""
        if self.annotations is not None:
            return np.where(np.array(self.annotations['symbol']) == 't')[0]
        else:
            return None

    def _get_intervals(self, indices):
        """Return interval for indices."""
        if self.annotations is not None:
            return [(self.annotations['sample'][index-1], self.annotations['sample'][index+1])
                    for index in indices]
        else:
            return None

    def _get_df(self):
        """Return waveforms and categorical labels as a DataFrame."""
        # Create time array
        time = np.expand_dims(np.arange(len(self.waveforms)) / self.fs, axis=1)

        # Create waveform DataFrame
        waveforms = pd.DataFrame(data=np.concatenate((time, self.waveforms, np.full(time.shape, np.nan)), axis=1),
                                 columns=['time', 'ch1', 'ch2', 'label'])

        # Add labels
        waveforms = self._add_labels(df=waveforms, intervals=self.p_wave_intervals, label='p')
        waveforms = self._add_labels(df=waveforms, intervals=self.qrs_wave_intervals, label='qrs')
        waveforms = self._add_labels(df=waveforms, intervals=self.t_wave_intervals, label='t')

        # Filter DataFrame to annotated interval
        waveforms = waveforms.iloc[waveforms.dropna().index[0]:waveforms.dropna().index[-1]].reset_index()

        # Add NaN label
        waveforms['label'][waveforms['label'].isnull()] = 'na'

        # Add integer labels
        waveforms['label_int'] = waveforms['label'].replace(self.label_dict)

        return waveforms

    @staticmethod
    def _add_labels(df, intervals, label):
        """Add labels to DataFrame"""
        for interval in intervals:
            df['label'][(df.index >= interval[0]) & (df.index <= interval[1])] = label
        return df

    def plot_waveform(self):
        """Plot waveform with labels."""
        # Setup figure
        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))

        # Plot channel 1
        ax1.plot(self.df['time'][self.df['label'] == 'p'], self.df['ch1'][self.df['label'] == 'p'],
                 '-', lw=1.5, color='r')
        ax1.plot(self.df['time'][self.df['label'] == 'qrs'], self.df['ch1'][self.df['label'] == 'qrs'],
                 '-', lw=1.5, color='g')
        ax1.plot(self.df['time'][self.df['label'] == 't'], self.df['ch1'][self.df['label'] == 't'],
                 '-', lw=1.5, color='b')
        ax1.plot(self.df['time'][self.df['label'] == 'na'], self.df['ch1'][self.df['label'] == 'na'],
                 '-', lw=1.5, color='k')
        # ax1.set_ylabel('Normalized Amplitude', fontsize=22)
        # ax1.set_xlim([0, time_series_filt_ts.max()])
        # ax1.tick_params(labelbottom='off')
        # ax1.yaxis.set_tick_params(labelsize=16)

        # Plot channel 2
        ax2.plot(self.df['time'], self.df['ch2'].values, '-k', lw=1.5)
        # ax2.set_xlabel('Time, seconds', fontsize=22)
        # ax2.set_ylabel('Class Activation Map', fontsize=22)
        # ax2.set_xlim([0, time_series_filt_ts.max()])
        # # ax2.set_ylim([cam_filt.min()-0.05, cam_filt.max()+0.05])
        # ax2.set_ylim([-3, 35])
        # ax2.xaxis.set_tick_params(labelsize=16)
        # ax2.yaxis.set_tick_params(labelsize=16)

        plt.show()
