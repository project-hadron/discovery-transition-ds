"""

A setuptools based setup module
with single-source versioning

See:
https://packaging.python.org/en/latest/distributing.html
https://packaging.python.org/guides/single-sourcing-package-version/

"""

import re
# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))


def read(*parts):
    filename = path.join(here, *parts)
    with open(filename, encoding='utf-8') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='discovery-transition-ds',
    version=find_version('ds_discovery', '__init__.py'),
    description='Advanced data cleaning, data wrangling and feature extraction tools for ML engineers',
    long_description=read('README.rst'),
    url='http://github.com/gigas64/discovery-transition-ds',
    author='Gigas64',
    author_email='gigas64@opengrass.net',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Adaptive Technologies',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='Wrangling ML Visualisation Dictionary Discovery Productize Classification Feature Engineering Cleansing',
    packages=find_packages(exclude=['tests', 'guides', 'data', 'jupyter']),
    license='BSD',
    include_package_data=True,
    package_data={
        # If any package contains *.yaml or *.csv files, include them:
        '': ['*.yaml', '*.csv'],
    },
    python_requires='>=3.7',
    install_requires=[
        'aistac-foundation>=2.12',
        'pandas>=1.1',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'boto3',
        'botocore',
        'fsspec',
        's3fs',
        'pyyaml',
    ],
    extras_require={},
    entry_points={},
    test_suite='tests',
)
