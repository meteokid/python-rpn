#!/bin/ksh93

d.compile -o libburp_c.a -src burp_api.c -O 0 -librmn rmn

# no rmn shared library available for pgi compiler
if [[ "${PUBLISH_ARCH}" != pgi* ]]; then 
    d.compile -o libburp_c_shared.so -src burp_api.c -librmn rmnshared_016.2
fi
