from numpy.distutils.core import setup, Extension
import os, distutils, string
import rpn_version

# architecture='Linux_pgi611'
architecture = os.getenv('EC_ARCH')
runtime_libs=['-Wl,-rpath,/usr/local/env/armnlib/lib/'+architecture]
SharedLd=distutils.sysconfig.get_config_vars('LDSHARED')
SharedLd=string.split(SharedLd[0])

print 'Debug architecture=',architecture
print 'Debug runtime_libs=',runtime_libs
print 'Shared Objects loaded with',SharedLd

Fstd_module = Extension('Fstdc',
            include_dirs = ['/usr/local/env/armnlib/include','/usr/local/env/armnlib/include/'+architecture],
            libraries = ['PyFTN_helpers','rmn_shared_beta10'],
            extra_objects = ['utils/get_corners_xy.o'],
            extra_link_args=runtime_libs,
            library_dirs = ['/usr/local/env/armnlib/lib/'+architecture],
            sources = ['Fstdc.c'])

jimc_module = Extension('jimc',
            include_dirs = ['/usr/local/env/armnlib/include','/usr/local/env/armnlib/include/'+architecture],
            libraries = ['PyFTN_helpers','rmn_shared_beta10'],
            extra_objects = [
                'jim/jim_grid_mod.o',
                'jim/jim_xch_halo_nompi.o',
                'utils/vect_mod.o'],
            extra_link_args=runtime_libs,
            library_dirs = ['/usr/local/env/armnlib/lib/'+architecture],
            sources = ['jimc.c'])

scripc_module = Extension('scripc',
            include_dirs = ['/usr/local/env/armnlib/include','/usr/local/env/armnlib/include/'+architecture,'./utils','./scrip'],
            libraries = ['PyFTN_helpers','rmn_shared_beta10'],
            extra_objects = [
                'scrip/kinds_mod.o',
                'scrip/constants.o',
                'scrip/grids.o',
                'scrip/remap_vars.o',
                'scrip/remap_distwgt.o',
                'scrip/remap_conserv.o',
                'scrip/remap_bilinear.o',
                'scrip/remap_bicubic.o',
                'scrip/remap_write.o',
                'scrip/scrip.o',
                'scrip/remap.o',
                'scrip/scrip_interface.o'],
            extra_link_args=runtime_libs,
            library_dirs = ['/usr/local/env/armnlib/lib/'+architecture],
            sources = ['utils/py_capi_ftn_utils.c','scripc.c'])


setup(name = 'rpnstd',
    version = rpn_version.__VERSION__,
    description = 'Python Interface to some ARMNLIB RPN STD files function',
    author = 'Mario Lepine',
    author_email = 'mario.lepine@ec.gc.ca',
    maintainer = 'Stephane Chamberland',
    maintainer_email = 'stephane.chamberland@ec.gc.ca',
    url = 'http://arxt20/~armnsch/arxt20/wiki/howto/pyhton-rpn',
    long_description = '''
Python Interface to some ARMNLIB RPN STD files function
Base Interfaces are defined in the Fstdc sub-package.
More python-esk clasess are defined in the rpnstd sub-package
''',
    py_modules=['rpn_version','rpn_helpers','rpnstd','jim','scrip'],
    ext_modules = [Fstd_module,jimc_module,scripc_module])

