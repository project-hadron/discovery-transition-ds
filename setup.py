# To use a consistent encoding
from codecs import open
import os

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


about = {}
with open(os.path.join(here, 'ds_discovery', '__version__.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)

with open('README.rst', 'r', 'utf-8') as f:
    readme = f.read()


setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
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
        'aistac-foundation',
        'pandas>',
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
