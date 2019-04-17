"""
qtdb.py
--------
This module provides classes and methods for creating the QT database.
By: Sebastian D. Goodfellow and Noel Kippers, 2019
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import wfdb

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
        self.record_ids = None

        # Create folders
        self._create_folders()

    def generate_db(self):
        """Generate raw and processed databases."""
        # Generate raw database
        self.generate_raw_db()

    def generate_raw_db(self):
        """Generate the raw version of the QT database in the 'raw' folder."""
        print('Generating Raw QT Database...\nSave path: {}'.format(self.raw_path))
        # Download database
        wfdb.dl_database(self.db_name, self.raw_path)

        # Get list of recordings
        self.record_ids = [file.split('.')[0] for file in os.listdir(self.raw_path) if '.dat' in file]
        print('Complete!\n')

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
