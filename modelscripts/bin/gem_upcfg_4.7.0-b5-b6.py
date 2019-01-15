#!/usr/bin/env python
import modelcfg,optparse,sys,os,re,shutil,math

def gem_upcfg_470b56(cfg):
    toremove = (
        ('gement' , 'e_schm_adcub'),
        ('gement' , 'e_schm_stlag'),
        ('gement' , 'Pil_hblen'),
        ('gement' , 'Pil_lancemodel'),
        ('gement' , 'topo_dgfmx_L'),
        ('gement' , 'topo_dgfms_L'),
        ('gement' , 'topo_filmx_L'),
        ('gement' , 'topo_clip_oro_L')
        )
    toconvert = (
        ('gem_cfgs' , ('init_dfnp','init_dflength_S','p')),
        ('gem_cfgs' , ('init_dfpl_8','init_dfpl_S','s')),
        ('gem_cfgs' , ('out3_postfreq','out3_postfreq_S','m')),
        ('step' , ('Step_rsti','Fcst_rstrt_S','p')),
        ('step' , ('Step_bkup','Fcst_bkup_S','p')),
        ('step' , ('Step_gstat','Fcst_gstat_S','p')),
        ('step' , ('Step_spinphy','Fcst_spinphy_S','p')),
        ('step' , ('Step_nesdt','Fcst_nesdt_S','s')),
        ('physics_cfgs' , ('kntrad','kntrad_S','p')),
        ('physics_cfgs' , ('shlcvt','pbl_shal',''))
       )
    # Add/modify value for key in settings file
    for (namelist,items) in toconvert:
        try:
            (oldname,newname,unit) = items
            value = cfg.get(oldname,namelist)
        except:
            sys.stdout.write('Ignore - not present: '+namelist+'/'+oldname+' => '+newname+'\n')
            continue
        if value:
            cfg.rm(oldname,namelist)
            if unit:
                try:
                    value = str(int(float(value)))
                except:
                    pass
                cfg.set(newname,"'"+value+unit+"'",namelist)
                sys.stdout.write('Convert: '+namelist+'/'+oldname+' => '+newname+'\n')
            else:
                cfg.set(newname,value,namelist)
                sys.stdout.write('Rename : '+namelist+'/'+oldname+' => '+newname+'\n')
        else:
            sys.stdout.write('Ignore - set error: '+namelist+'/'+oldname+' => '+newname+'\n')
            
    # Remove keys in settings file
    for (namelist,oldname) in toremove:
        try:
            value = cfg.get(oldname,namelist)
        except:
            sys.stdout.write('Ignore - not present: '+namelist+'/'+oldname+' => '+newname+'\n')
            continue
            if value:
                try:
                    cfg.rm(oldname,namelist)
                    sys.stdout.write('Remove : '+namelist+'/'+oldname+'\n')
                except:
                    sys.stdout.write('Ignore - rm error: '+namelist+'/'+oldname+'\n')
                    continue
            else:
                sys.stdout.write('Ignore - no value: '+namelist+'/'+oldname+'\n')
            

if __name__ == "__main__":
    
    # Command line arguments
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-f","--file",dest="nml_file",default="./gem_settings.nml",
                      help="Name of FILE containing namelist [default 'gem_settings.nml']",metavar="FILE")
    parser.add_option("-b","--backup",dest="backup",action="store_true",
                      help="Create a backup (.bkp) of the input FILE",metavar="FILE")
    (options,args) = parser.parse_args()

    # Create backup if requested
    if options.backup:
        bkpfile = options.nml_file+'.bkp'
        try:
            shutil.copy(options.nml_file,bkpfile)
            sys.stdout.write('Worte a backup file: '+bkpfile+'\n')
        except IOError:
            sys.stderr.write('Aborting because requested backup cannot be created\n')
            raise

    # Retrieve namelist entries
    sys.stdout.write('Reading nml file: '+options.nml_file+'\n')
    cfg = modelcfg.Settings(file=options.nml_file)

    gem_upcfg_470b56(cfg)

    # Write modified settings file
    try:
        sys.stdout.write('Writing updated nml file: '+options.nml_file+'\n')
        cfg.write(options.nml_file)
    except IOError:
        sys.stderr.write('Cannot write to '+options.nml_file+'\n')
        raise
