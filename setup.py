#!/usr/bin python3
# -*- coding: utf-8 -*-

import re
from os import path
from codecs import open
from setuptools import setup
from pkg_resources import parse_version

VERSIONFILE = "vafnet/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="vafnet",
    version=__version__,
    license="Apache License 2.0",
    author="Max W. Y. Lam",
    author_email="maxingaussian@gmail.com",
    description="Variational Activation Functions for Deep Learning",
    long_description=long_description,
    keywords="machine-learning deep-learning neural-networks gaussian-process",
    url="https://github.com/MaxInGaussian/VAFnet",
    download_url='https://github.com/MaxInGaussian/VAFnet/tarball/'+__version__,
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    packages=["vafnet"],
    package_dir={'vafnet': 'vafnet'},
    py_modules=['vafnet.__init__'],
    install_requires=['numpy>=1.7',
                      'six>=1.10.0'],
    extras_require={
        'edward': ['edward>=1.3.4'],
        'tensorflow': ['tensorflow>=1.2.0rc0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.2.0rc0'],
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': ['matplotlib>=1.3']},
    tests_require=['pytest', 'pytest-pep8'],
)