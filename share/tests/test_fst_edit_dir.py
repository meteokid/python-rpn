'Test fst_edit_dir'

def main():
    'Copy FSTD file, then change time-step number'

    from shutil import copy
    from os import chmod, stat
    from stat import S_IWUSR

    from rpnpy.librmn.const import FST_RO, FST_RW
    from rpnpy.librmn.fstd98 import fstcloseall, fstinf, fstopenall, fst_edit_dir

    ref_file = '/users/dor/armn/env/SsmBundles/GEM/d/gem-data/gem-data_4.2.0/gem-data_4.2.0_all/share/data/dfiles/bcmk/2009042700_012'

    fref = fstopenall(ref_file, FST_RO)
    [dateo, npas] = read_dateo_npas(fref)
    fstcloseall(fref)

    print 'Reference file'
    print 'dateo: ' + str(dateo)
    print 'npas: ' + str(npas)

    new_file = 'new.fst'

    copy(ref_file, new_file)
    chmod(new_file, stat(new_file).st_mode | S_IWUSR)

    fnew = fstopenall(new_file, FST_RW)
    [dateo, npas] = read_dateo_npas(fnew)

    print 'New file'
    print 'Before fst_edit_dir'
    print 'dateo: ' + str(dateo)
    print 'npas: ' + str(npas)

    key = fstinf(fnew, nomvar='UU')['key']
    fst_edit_dir(key, npas=49)

    [dateo, npas] = read_dateo_npas(fnew)

    print 'After fst_edit_dir'
    print 'dateo: ' + str(dateo)
    print 'npas: ' + str(npas)

    fstcloseall(fnew)

def read_dateo_npas(iunit):
    'Read date of original analysis and time-step number'
    from rpnpy.librmn.fstd98 import fstinf, fstprm

    key = fstinf(iunit, nomvar='UU')['key']

    dateo = fstprm(key)['dateo']
    npas = fstprm(key)['npas']

    return dateo, npas

if __name__ == '__main__':
    main()
