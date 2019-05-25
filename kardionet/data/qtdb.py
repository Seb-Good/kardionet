"""This module provides a class and methods for extracting the QT database.

By: Sebastian D. Goodfellow and Noel Kippers, 2019
"""
import os
import logging

import wfdb

from kardionet import DATA_DIR
from kardionet.data.record import Record

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class QTDB(object):
    """The QT Database.

    https://physionet.org/physiobank/database/qtdb/
    """

    def __init__(self, data_dir=DATA_DIR):
        """Init QTDB."""
        self.db_name = 'qtdb'
        self.raw_path = os.path.join(data_dir, 'raw')
        self.processed_path = os.path.join(data_dir, 'processed')
        self.record_names = None
        self._create_folders()

    def _create_folders(self):
        """Create a folders for raw and processed data."""
        _create_directory(directory=self.raw_path)
        _create_directory(directory=self.processed_path)

    def generate_db(self):
        """Generate raw and processed databases."""
        self.generate_raw_db()
        self.generate_processed_db()

    def generate_raw_db(self):
        """Generate the raw version of the QT database in the 'raw' folder."""
        LOGGER.info('Generating Raw QT Database...\nSave path: {}'.format(self.raw_path))
        wfdb.dl_database(self.db_name, self.raw_path)
        self.record_names = self._get_record_names()
        LOGGER.info('Complete!')

    def generate_processed_db(self):
        """Generate the processed version of the QT database in the 'processed' folder."""
        self.record_names = self._get_record_names()
        for i, record_name in enumerate(self.record_names):
            record = Record(record_name=record_name)
            record.save_csv()
            LOGGER.info('Processed: {}/{} wfdb.rdrecord'.format(i+1, len(self.record_names)))

    def _get_record_names(self):
        """Return list of records in 'raw' path."""
        return [f.split('.')[0] for f in os.listdir(self.raw_path) if '.dat' in f]


def _create_directory(directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, '.gitignore'), 'w') as f:
        f.write('*\n!.gitignore')
