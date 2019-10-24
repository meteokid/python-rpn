import sys
import ctypes as _ct

if sys.version_info < (3,):
    integer_types = (int, long,)
    range = xrange
else:
    integer_types = (int,)
    long = int
    range = range

C_STRSETLEN = lambda s,l: "{{:{}s}}".format(l).format(s[:l])
C_STRSETLEN.__doc__ = 'Return str with specified len, cut extra right char or right pad with spaces'

def C_WCHAR2CHAR(x, l=None):
    """
    Convert str to bytes
    """
    s = str(x) if l is None else C_STRSETLEN(str(x), l)
    try:
        return bytes(s.encode('ascii'))
    except UnicodeEncodeError:
        return bytes(s.encode('utf-8'))
C_WCHAR2CHARL = C_WCHAR2CHAR

def C_CHAR2WCHAR(x, l=None):
    """
    Convert bytes to str
    """
    try:
        s = str(x.decode('ascii'))
    except UnicodeDecodeError:
        s = x.decode('utf-8')
    return s if l is None else C_STRSETLEN(s, l)
C_CHAR2WCHARL = C_CHAR2WCHAR

C_WCHAR2CHAR_COND = lambda x: x if isinstance(x, bytes) else C_WCHAR2CHAR(x)
C_WCHAR2CHAR_COND.__doc__ = 'Conditionnal Convert str to bytes'
C_CHAR2WCHAR_COND = lambda x: C_CHAR2WCHAR(x) if isinstance(x, bytes) else x
C_CHAR2WCHAR_COND.__doc__ = 'Conditionnal Convert bytes to str'

C_MKSTR = lambda x: _ct.create_string_buffer(C_WCHAR2CHAR(x))
C_MKSTR.__doc__ = 'alias to ctypes.create_string_buffer, make sure bytes are provided'
