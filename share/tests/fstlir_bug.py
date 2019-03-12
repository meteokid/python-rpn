from os import environ
from os.path import join

import rpnpy.librmn.all as rmn
print 'librmn=',rmn.librmn

data_file = join(environ['CMCGRIDF'], 'prog', 'gsloce', '2015070706_042')

funit = rmn.fstopenall(data_file,rmn.FST_RO)
#r = rmn.fstlir(funit, nomvar='UUW', typvar='P@')
k = rmn.fstinf(funit, nomvar='UUW', typvar='P@')
## print k
## print rmn.fstprm(k['key'])
r = rmn.fstluk(k['key'])

l = rmn.fstinl(funit)
print 'data_file=',data_file,len(l)
for k in l:
    p = rmn.fstprm(k)
    print p['nomvar'], p['typvar'], p['datyp']
    if p['datyp'] != 1344:
        r = rmn.fstluk(k)
        del(r['d'])
        del r
rmn.fstcloseall(funit)
