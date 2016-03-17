import sys

if sys.version_info < (3,):
    integer_types = (int, long,)
else:
    integer_types = (int,)
    long = int
