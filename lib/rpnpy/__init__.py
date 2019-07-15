import sys
import ctypes as _ct

if sys.version_info < (3,):
    integer_types = (int, long,)
    range = xrange
else:
    integer_types = (int,)
    long = int
    range = range

C_WCHAR2CHAR = lambda x: bytes(str(x).encode('ascii'))
C_WCHAR2CHAR.__doc__ = 'Convert str to bytes'

C_CHAR2WCHAR = lambda x: str(x.decode('ascii'))
C_CHAR2WCHAR.__doc__ = 'Convert bytes to str'

C_MKSTR = lambda x: _ct.create_string_buffer(C_WCHAR2CHAR(x))
C_MKSTR.__doc__ = 'alias to ctypes.create_string_buffer, make sure bytes are provided'
