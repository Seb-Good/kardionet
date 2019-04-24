"""Tests for QTDB class."""
import unittest

from kardionet.data.qtdb import QTDB


class TestQTDB(unittest.TestCase):
    """Tests for QTDB."""

    def test__init__(self):
        """Test QTDB initialization."""
        _qtdb = QTDB()
        self.assertIsInstance(_qtdb, QTDB)


if __name__ == '__main__':
    unittest.main()
