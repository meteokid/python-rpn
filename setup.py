from numpy.distutils.core import setup, Extension
import os, distutils, string

# architecture='Linux_pgi611'
architecture = os.getenv('EC_ARCH')
print 'Debug architecture=',architecture
runtime_libs=['-Wl,-rpath,/usr/local/env/armnlib/lib/'+architecture]
print 'Debug runtime_libs=',runtime_libs
SharedLd=distutils.sysconfig.get_config_vars('LDSHARED')
SharedLd=string.split(SharedLd[0])
print 'Shared Objects loaded with',SharedLd

Fstd_module = Extension('Fstdc',
                    include_dirs = ['/usr/local/env/armnlib/include','/usr/local/env/armnlib/include/'+architecture],
                    libraries = ['PyFTN_helpers','rmn_shared_beta10'],
                    extra_link_args=runtime_libs,
                    library_dirs = ['/usr/local/env/armnlib/lib/'+architecture],
                    sources = ['Fstdc.c'])

setup (name = 'truc',
      version = '1.0',
      ext_modules = [Fstd_module])
