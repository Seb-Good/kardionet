"""Package wide settings."""
import os
import logging

logging.basicConfig()

HEAD_DIR = (
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

DATA_DIR = os.path.join(HEAD_DIR, 'data')
