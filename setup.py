from setuptools import setup, find_packages
import sys
from glob import glob
import os

# Build version file.
from subprocess import check_call
versionfile = os.path.join('lib','rpnpy','version.py')
makefile = os.path.join('include','Makefile.local.rpnpy.mk')
if os.path.exists(makefile):
  if os.path.exists(versionfile):
    os.remove(versionfile)
  check_call(['make','-f','include/Makefile.local.rpnpy.mk','rpnpy_version.py'], env={'rpnpy':'.'})

# Add './lib' to the search path, so we can access the version info.
sys.path.append('lib')
from rpnpy.version import __VERSION__

# If the shared library source is available (for librmn, etc.)
# then build the shared libraries and bundle them here.
if os.path.exists(os.path.join('lib','rpnpy','_sharedlibs','librmn','Makefile')):
  from rpnpy._sharedlibs import get_extra_setup_args
  extra_setup_args = get_extra_setup_args('rpnpy','_sharedlibs')
else:
  extra_setup_args = {}


setup (
  name = 'eccc_rpnpy',
  version = __VERSION__,
  description = 'A Python interface for the RPN libraries at Environment and Climate Change Canada',
  long_description = open('DESCRIPTION').read(),
  url = 'https://github.com/meteokid/python-rpn',
  author = 'Stephane Chamberland',
  license = 'LGPL-2.1',
  keywords = 'rpnpy python-rpn vgrid libdescrip librmn rmnlib',
  packages = find_packages('lib'),
  py_modules = ['Fstdc','rpn_helpers','rpnstd'],
  scripts = glob('bin/rpy.*'),
  package_dir = {'':'lib'},
  install_requires = ['numpy','pytz'],
  **extra_setup_args
)
