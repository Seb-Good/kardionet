"""
config.py
---------
Config variables for the project.
By: Sebastian D. Goodfellow, Ph.D., Noel Kippers, Ph.D., 2019
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# Local imports
import os

# Set working directory
WORKING_DIR = (
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__)
            )
        )
    )
)

# Set data directory
DATA_DIR = os.path.join(WORKING_DIR, 'data')
