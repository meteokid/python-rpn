import rpnpy.librmn.all as rmn
data_file = '/cnfs/dev/cmd/cmde/afsgbla/watroute2py/watroute/budget/io/test/input_test.fst'
funit = rmn.fstopenall(data_file,rmn.FST_RO)
k = rmn.fstinf(funit, nomvar='I1D')
r = rmn.fstlir(funit, nomvar='I1D')
print k
print r
rmn.fstcloseall(funit)
