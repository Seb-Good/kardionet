"""Setup for KardioNet"""
from setuptools import setup, find_packages

setup(
    name='kardionet',
    version='0.0.1',
    description='A package to automatically segment ecg waveforms.',
    url='https://github.com/Seb-Good/kardionet.git',
    author='Sebastian D. Goodfellow and Ph.D., Noel Kippers, Ph.D.',
    license='MIT',
    keywords='deep learning',
    zip_safe=False,
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    include_package_data=True,
    install_requires=[],
    setup_requires=[],
    tests_require=[],
    entry_points={'console_scripts': ['kardionet=kardionet.__main__:cli'], },
)
