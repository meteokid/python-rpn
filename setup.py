from numpy.distutils.core import setup, Extension
import os, distutils, string
import rpn_version

#TODO: need rmnlib-dev

#myecarch = 'Linux_pgi611'
#myrmnlib = 'rmn_shared_beta10'
#myecarch = 'Linux_pgi9xx'
myecarch = 'Linux_x86-64/pgi9xx'
myrmnlib = ['PyFTN_helpers','rmnbetashared_011']

eclibpath = os.getenv('EC_LD_LIBRARY_PATH')
ecincpath = os.getenv('EC_INCLUDE_PATH')
architecture = os.getenv('EC_ARCH')

eclibsharedpath = ''
for mypath in eclibpath.split():
    isok = True
    for item in myrmnlib:
        if not os.path.exists(mypath+'/lib'+item+'.so'):
            isok = False
    if isok:
        eclibsharedpath = mypath

if not architecture == myecarch:
    print("WARNING: EC_ARCH should be "+myecarch+" and is: "+architecture)
    #TODO: stop
if not eclibsharedpath:
    print("WARNING: Could not find LIB PATH for "+str(myrmnlib))
    #TODO: stop

#TODO: FIND inc path (Not in EC_INCLUDE_PATH yet), using old $ARMNLIB/includefor now

runtime_libs=['-Wl,-rpath,'+eclibsharedpath]
SharedLd=distutils.sysconfig.get_config_vars('LDSHARED')
SharedLd=string.split(SharedLd[0])

print 'Debug architecture=',architecture
print 'Debug runtime_libs=',runtime_libs
print 'Shared Objects loaded with',SharedLd

Fstd_module = Extension('Fstdc',
            include_dirs = ['/usr/local/env/armnlib/include','/usr/local/env/armnlib/include/'+architecture,'./utils'],
            libraries = myrmnlib,
            extra_objects = ['utils/get_corners_xy.o'],
            extra_link_args = runtime_libs,
            library_dirs = [eclibsharedpath],
            sources = ['utils/py_capi_ftn_utils.c','Fstdc.c'])

setup(name = 'rpnstd',
    version = rpn_version.__VERSION__,
    description = 'Python Interface to some ARMNLIB RPN STD files function',
    author = 'Stephane Chamberland, Mario Lepine, Michel Valin',
    author_email = 'stephane.chamberland@ec.gc.ca',
    maintainer = 'Stephane Chamberland',
    maintainer_email = 'stephane.chamberland@ec.gc.ca',
    url = 'https://wiki.cmc.ec.gc.ca/wiki/User:Chamberlands/Projects/Python-RPN',
    long_description = '''
Python Interface to some ARMNLIB RPN STD files function
Base Interfaces are defined in the Fstdc sub-package.
More python-esque classes are defined in the rpnstd sub-package
''',
    py_modules=['rpn_version','rpn_helpers','rpnstd'],
    ext_modules = [Fstd_module])

