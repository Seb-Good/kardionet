"""Setup for KardioNet"""
from setuptools import setup

setup(
    name='kardionet',
    version='0.0.1',
    description='A package to automatically segment ecg waveforms.',
    url='https://github.com/Seb-Good/kardionet.git',
    author='Sebastian D. Goodfellow, Ph.D., Noel Kippers, Ph.D.',
    license='MIT',
    keywords='deep learning',
    package_dir={'': 'kardionet'},
    zip_safe=False,
)
