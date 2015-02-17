from numpy.distutils.core import setup, Extension
import os, distutils, string
import rpn_version

#TODO: . s.ssmuse.dot legacy pgi9xxshared rmnlib-dev

myecarch = ['Linux_x86-64/intel13sp1u2']
myrmnlib = [os.getenv('RMNLIBSHARED','rmnshared_015.1'),]

architecture = os.getenv('EC_ARCH')
eclibpath = os.getenv('EC_LD_LIBRARY_PATH')
ecincpath = os.getenv('EC_INCLUDE_PATH')+' '+'./utils'

eclibsharedpath = ''
for mypath in eclibpath.split():
    isok = True
    for item in myrmnlib:
        if not os.path.exists(mypath+'/lib'+item+'.so'):
            isok = False
    if isok:
        eclibsharedpath = mypath

if not architecture in myecarch:
    print("WARNING: EC_ARCH should be "+' or '.join(myecarch)+" and is: "+architecture)
    #TODO: stop
if not eclibsharedpath:
    print("WARNING: Could not find LIB PATH for "+str(myrmnlib))
    #TODO: stop

runtime_libs=[os.getenv('LDFLAGS',''),'-Wl,-rpath,'+eclibsharedpath]
SharedLd=distutils.sysconfig.get_config_vars('LDSHARED')
SharedLd=string.split(SharedLd[0])

print '#Debug architecture=',architecture
print '#Debug runtime_libs=',runtime_libs
print '#Debug Shared Objects loaded with',SharedLd
print '#Debug eclibpath=',eclibpath
print '#Debug eclibshar=',eclibsharedpath
print '#Debug ecincpath=',ecincpath

Fstd_module = Extension('Fstdc',
            include_dirs = ecincpath.split(' '),
            libraries = myrmnlib,
            extra_objects = ['utils/get_corners_xy.o'],
            extra_link_args = runtime_libs,
            extra_compile_args = ['-Wtrigraphs','-fpic','-fp-model precise','-mkl'],
            library_dirs = [eclibsharedpath],
            sources = ['utils/py_capi_ftn_utils.c','Fstdc.c','utils/numpy_compat.c'])

setup(name = 'rpnstd',
    version = rpn_version.__VERSION__,
    description = 'Python Interface to some ARMNLIB RPN STD files function',
    author = 'Stephane Chamberland, Mario Lepine, Michel Valin, Christopher Subich',
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

