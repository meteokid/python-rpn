from numpy.distutils.core import setup, Extension
import os, distutils, string
import rpn_version

# architecture='Linux_pgi611'
architecture = os.getenv('EC_ARCH')
if not architecture == 'Linux_pgi611':
    print("WARNING: EC_ARCH should be Linux_pgi611 and is: "+architecture)

runtime_libs=['-Wl,-rpath,/usr/local/env/armnlib/lib/'+architecture]
SharedLd=distutils.sysconfig.get_config_vars('LDSHARED')
SharedLd=string.split(SharedLd[0])

print 'Debug architecture=',architecture
print 'Debug runtime_libs=',runtime_libs
print 'Shared Objects loaded with',SharedLd

Fstd_module = Extension('Fstdc',
            include_dirs = ['/usr/local/env/armnlib/include','/usr/local/env/armnlib/include/'+architecture,'./utils'],
            libraries = ['PyFTN_helpers','rmn_shared_beta10'],
            extra_objects = ['utils/get_corners_xy.o'],
            extra_link_args=runtime_libs,
            library_dirs = ['/usr/local/env/armnlib/lib/'+architecture],
            sources = ['utils/py_capi_ftn_utils.c','Fstdc.c'])

setup(name = 'rpnstd',
    version = rpn_version.__VERSION__,
    description = 'Python Interface to some ARMNLIB RPN STD files function',
    author = 'Stephane Chamberland, Mario Lepine',
    author_email = 'stephane.chamberland@ec.gc.ca',
    maintainer = 'Stephane Chamberland',
    maintainer_email = 'stephane.chamberland@ec.gc.ca',
    url = 'https://wiki.cmc.ec.gc.ca/wiki/User:Chamberlands/Projects/Python-RPN',
    long_description = '''
Python Interface to some ARMNLIB RPN STD files function
Base Interfaces are defined in the Fstdc sub-package.
More python-esk classes are defined in the rpnstd sub-package
''',
    py_modules=['rpn_version','rpn_helpers','rpnstd'],
    ext_modules = [Fstd_module])

