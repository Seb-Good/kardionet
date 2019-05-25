"""Tests for QTDB class."""
import os
import unittest
import shutil

from kardionet.data.qtdb import QTDB

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')


class TestQTDB(unittest.TestCase):
    """Tests for QTDB."""

    def tearDown(self):
        """Will remove the TEST_DATA_DIR after test completes."""
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

    def test__init__(self):
        """Test QTDB initialization."""
        _qtdb = QTDB(data_dir=TEST_DATA_DIR)
        self.assertIsInstance(_qtdb, QTDB)
        self.assertTrue('processed' in os.listdir(TEST_DATA_DIR))
        self.assertTrue('raw' in os.listdir(TEST_DATA_DIR))


if __name__ == '__main__':
    unittest.main()
