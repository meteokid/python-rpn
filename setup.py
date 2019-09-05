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
)
