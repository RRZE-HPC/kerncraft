#!/usr/bin/env python
from __future__ import absolute_import
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist
from distutils.dir_util import mkpath
from codecs import open  # To use a consistent encoding
import sys, os

here = os.path.abspath(os.path.dirname(__file__))

# Stolen from pycparser
def _run_build_tables(dir):
    targetdir = os.path.join(dir, 'kerncraft', 'pycparser')
    # mkpath is a distutils helper to create directories
    mkpath(targetdir)
    from subprocess import call
    call([sys.executable, '_build_tables.py'], cwd=targetdir)


# Stolen and modified from pycparser
class build_py(_build_py):
    def run(self):
        # honor the --dry-run flag
        if not self.dry_run:
            # FIXME can we also build at the target dir?
            self.execute(_run_build_tables, (os.getcwd(),),
                         msg="Build the lexing/parsing tables")
        _build_py.run(self)


# Stolen and modified from pycparser
class sdist(_sdist):
    def make_release_tree(self, base_dir, files):
        self.execute(_run_build_tables, (os.getcwd(),),
                     msg="Build the lexing/parsing tables")
        dir = os.path.join('kerncraft', 'pycparser')
        files.append(os.path.join(dir, 'yacctab.py'))
        files.append(os.path.join(dir, 'lextab.py'))
        _sdist.make_release_tree(self, base_dir, files)

# Get the long description from the relevant file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kerncraft',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.4.0',

    description='Loop Kernel Analysis and Performance Modeling Toolkit',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/cod3monk/kerncraft',

    # Author details
    author='Julian Hammer',
    author_email='julian.hammer@fau.de',

    # Choose your license
    license='AGPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Utilities',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU Affero General Public License v3',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],

    # What does your project relate to?
    keywords='hpc performance benchmark analysis',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed. For an analysis of "install_requires" vs pip's

    # what is not found or up-to-date on pypi we get from github:
    #dependency_links = ['https://github.com/sympy/sympy/tarball/master#egg=sympy-0.7.7.dev0'],

    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'ruamel.yaml',
        'six',
        'sympy>=0.7.7',
        'pycachesim>=0.1.4',
        'pylru',
        'numpy',
        'pycparser>=2.14',
    ],

    # List additional groups of dependencies here (e.g. development dependencies).
    # You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'plot': ['matplotlib'],
        'test': ['requests'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'kerncraft': ['headers/dummy.c', 'headers/kerncraft.h', 'README.rst', 'LICENSE',
                      'pycparser/*.cfg'],
        'examples': [
            'machine-files/*.yaml',
            'kernels/*.c',
            'kernels/*.testcases'],
        'tests': ['test_files/*.c', 'test_files/*.yaml', '*.py'],
    },
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages.
    # see http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'kerncraft=kerncraft.kerncraft:main',
            'iaca_marker=kerncraft.iaca_marker:main',
            'likwid_bench_auto=kerncraft.likwid_bench_auto:main',
            'picklemerge=kerncraft.picklemerge:main',
            'cachetile=kerncraft.cachetile:main'
        ],
    },
    
    cmdclass={'build_py': build_py, 'sdist': sdist},
)
