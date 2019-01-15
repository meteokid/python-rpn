#!/bin/ksh

myself=$0
myecho="echo -e"
[[ "x$(echo -e)" != "x" ]] && myecho="echo"

DESC='Run GEM test cases interactively'
USAGE="USAGE: ${myself##*/} [-h] [--nompi | [--ptopo=NPEXxNPEYxNOMP [--btopo=NPEXxNPEY]] [--submit=MACH [-t TIME] [--smt] [--dir=DRINAME]] [--make] [--stdout] [-c CASENAME | -l] [--debug] [--nocheck]"

fromversion="" #4.8.a10
myversion=${fromversion}
doupcfg=0
#TODO: lam2 cases, reduce dt and increase Grd_maxcfl to get MPI repro
#TODO: on AIX* test openmpi running and reproducibility of 2 different omp (do auto compare)
#TODO: on AIX* test mpi running and reproducibility of 2 different npx*npy (do auto compare)

#export PATH=${ATM_MODEL_BINDIR}:$PATH

#---- Functions ------------------------------------------------------

mytext=""
[[ x$BASE_ARCH == xLinux_x86-64 ]] && export mytext="-a"
export mydiff="diff $mytext"
export mygrep="egrep $mytext"

#====
usage_long() {
   toto=$($myecho $USAGE)
   more <<EOF
$DESC

$toto

Options:
    -h, --help    : print this help
    --nompi       : run w/o mpi (must be compile w/ nompi option
    --ptopo=9x9x9 : force proc topology to NPEXxNPEYxNOMP
    --btopo=9x9   : force i/o topology to NPEXxNPEY
    --dir=DRINAME : subdirname where to run the job
    --make        : build GEM abs (defautl is not to make)
    --stdout      : send model output to stdout instead of log file
    --valgrind    : run with valgrind
    --debug       : run gem in gdb
    -c CASENAME   : run model with config cfg_CASENAME (default is all)
    -l            : list avail cfg    --nocheck     : skip GEM version check

    --smt         : use SMT (to do)
    --submit=MACH : sumbit on MACH (to do)
    -t TIME       : sumbit time (to do)

EOF
}

myupcfg() {
   _dircfg=${1:-/dev/null}
   _here=$(pwd)
   cd ${_dircfg}/cfg_0000
   [[ x$doupcfg == x1 ]] && gem_upcfg -t nml --from ${fromversion} --to ${myversion}
   cd ${_here}
}


runonecase() {
   _dircfg=$1
   _ptopo=$2
   _btopo=$3
   _name=$4
   _runpil=${5:-dorunpil}

   hasranonecase=1

   if [[ ! -d ${_dircfg}/cfg_0000 ]] ; then
      cat <<EOF
WARNING: config dir not found: ${_dircfg}/cfg_0000
         Skipping ${_dircfg}
EOF
      return
   fi

   dirorig=$(pwd)
   baserundir=$(pwd)/$(rdevar build/run)

   cd ${baserundir}
   ln -s ${dirorig}/${_dircfg} . 2>/dev/null || true
   ln -s ${dirorig}/main*_${BASE_ARCH}*.Abs . 2>/dev/null || true

   if [[ x$stdout == x0 ]] ; then
      myupcfg ${_dircfg} >/dev/null
   else
      myupcfg ${_dircfg}
   fi
   
   if [[ x$nompi != x ]] ; then
      _ptopo=1x1x1
      _btopo=1x1
   else
      if [[ x$ptopo != x ]] ; then
         _ptopo=$ptopo
      fi
      npex=$(echo $_ptopo | cut -dx -f1) ; [[ x$npex == x ]] && npex=1
      npey=$(echo $_ptopo | cut -dx -f2) ; [[ x$npey == x ]] && npey=1
      nomp=$(echo $_ptopo | cut -dx -f3) ; [[ x$nomp == x ]] && nomp=1
      _ptopo=${npex}x${npey}x${nomp}
      if [[ x$btopo != x ]] ; then
         nblx=$(echo $btopo | cut -dx -f1) ; [[ x$nblx == x ]] && nblx=1
         nbly=$(echo $btopo | cut -dx -f2) ; [[ x$nbly == x ]] && nbly=1
         _btopo=${nblx}x${nbly}
      else
         if [[ x$_btopo == x ]]  ; then
            _btopo=${npex}x${npey}
         else
            nblx=$(echo $_btopo | cut -dx -f1) ; [[ x$nblx == x ]] && nblx=1
            nbly=$(echo $_btopo | cut -dx -f2) ; [[ x$nbly == x ]] && nbly=1
            _btopo=${nblx}x${nbly}
         fi
      fi
   fi

   if [[ x$submit != x ]] ; then
      echo "ERROR: submit not yet supported"
      return
   fi
   
   _theoc=''
   if [[  x${_runpil} == xtheoc ]] ; then
      # rm -rf RUNENT/*
      rm -rf RUNMOD/*
      _theoc='-theoc'
   fi

   echo '+ runonecase' $_dircfg $_ptopo $_btopo $_name $_runpil
   export OMP_NUM_THREADS=$nomp

   # if [[ x${_runpil} == xdorunpil ]] ; then 
   #    rm -rf RUNENT/*
   #    rm -rf RUNMOD/*
   #    cmd="Um_runent.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}"
   #    if [[ x$stdout == x0 ]] ; then
   #       logfile=${dirorig}/runent_${_name}_p${_ptopo}_b${_btopo}_${BASE_ARCH}.log
   #       $cmd > $logfile 2>&1
   #       echo "runent $($mygrep _status $logfile) $($mygrep GEMNTR $logfile | $mygrep GMT | cut -c16-60 | sort -u)"
   #    else
   #       $cmd
   #    fi
   # fi

   cmd="runprep -dircfg ${_dircfg}"
   if [[ x$stdout == x0 ]] ; then
      logfile=${dirorig}/runprep_${_name}_p${_ptopo}_b${_btopo}_${BASE_ARCH}.log
      $cmd >$logfile 2>&1
   else
      $cmd
   fi

   #cmd="Um_runmod.ksh -dircfg ${_dircfg} -ptopo ${_ptopo} -out_block ${_btopo} $nompi ${_theoc} -inorder"
   #cmd="Um_runmod.ksh -dircfg ${_dircfg} -ptopo ${_ptopo} -out_block ${_btopo} $nompi ${_theoc} -debug"
   #cmd="Um_runmod.ksh -dircfg ${_dircfg} -ptopo ${_ptopo} -out_block ${_btopo} $nompi ${_theoc}"
   cmd="Um_runmod.ksh -dircfg ${_dircfg} -ptopo ${_ptopo} $nompi ${_theoc}"

   if [[ x$stdout == x0 ]] ; then
      logfile=${dirorig}/runmod_${_name}_p${_ptopo}_b${_btopo}_${BASE_ARCH}.log
      logfile2=${dirorig}/blocstat_${_name}_p${_ptopo}_b${_btopo}_${BASE_ARCH}.log
      $cmd -inorder >$logfile 2>&1
      echo "runmod $($mygrep _status $logfile) $($mygrep GEMDM $logfile | $mygrep GMT | cut -c16-60 | sort -u)"
      $mygrep '(BLOC STAT|Min:|Std:)' $logfile | grep -v Memory > $logfile2
   else
      $cmd
   fi

   cd ${dirorig}

}

savepilotoutput() {
   _dircfg=${1:-/dev/null}
   _here=$(pwd)

   echo '+ savepilotoutput'

   cd RUNMOD_output
   rm -rf input
   mkdir input
   find . -name '3df*' -exec mv {} input \;
   cd ${_here}
   mv ${_dircfg}/cfg_0000/configexp.cfg ${_dircfg}/cfg_0000/configexp.cfg-0
   cat ${_dircfg}/cfg_0000/configexp.cfg-0 | $mygrep -v UM_EXEC_model_input \
      > ${_dircfg}/cfg_0000/configexp.cfg
   cat >> ${_dircfg}/cfg_0000/configexp.cfg <<EOF
UM_EXEC_model_input=$(pwd)/RUNMOD_output/input
EOF
}

#---- Inline Options -------------------------------------------------

domake=0
mycase=all
stdout=0
nompi=''
ptopo=''
btopo=''
dolist=0
previous=""
submit=
time=900
smt=
mydir=
valgrind=
debug=
docheckver=1

while [[ $# -gt 0 ]] ; do
   case $1 in
      (-h|--help) usage_long; exit 0;;
      (--nompi) nompi='-nompi';;
      (--smt) smt='-smt';;
      (--make) domake=1;;
      (--ptopo) [[ $# -gt 1 ]] && ptopo=$2 && shift ;;
      (--ptopo=*) ptopo=${1##*=} ;;
      (--btopo) [[ $# -gt 1 ]] && btopo=$2 && shift ;;
      (--btopo=*) btopo=${1##*=} ;;
      (-l|--list) dolist=1;;
      (--stdout) stdout=1;;
      (--valgrind) valgrind="valgrind";;
      (--debug) debug=-debug;;
      (--nocheck) docheckver=0;;
      (-c|--case) [[ $# -gt 1 ]] && mycase=$2 && shift ;;
      -*) echo "ERROR: Unrecognized option $1"
         echo -e $USAGE
         echo "---- ABORT ----"
         exit 1;;
      *)  echo "ERROR: Unrecognized option $1"
         echo -e $USAGE
         echo "---- ABORT ----"
         exit 1;;
   esac
   previous=$1
   shift
done

#---------------------------------------------------------------------

if [[ x$dolist == x1 ]] ; then
   ls -1d cfg_* | sed 's/cfg_//'
   exit 0
fi

if [[ x$docheckver == x1 ]] ; then 
if [[ x$ATM_MODEL_VERSION != x${myversion} ]] ; then
   cat <<EOF
ERROR: GEM version $myversion requested (${ATM_MODEL_VERSION:-NONE} loaded)
---- ABORT ----
EOF
   exit 1
    #. s.ssmuse.dot GEM/tests/${myversion}
fi
fi

if [[ x$domake == x1 ]] ; then
   echo "ERROR: domake option not avail" 1>&2
   exit 1
fi



hasranonecase=0

if [[ x$mycase == xtestcases8 ]] ; then
   name=RI
   runonecase cfg_$name 2x2x1 1x1 $name
   # [[ x$ptopo == x ]] && \
   #    runonecase cfg_$name 1x2x1 1x1 $name

   name=RDPS
   runonecase cfg_$name 4x2x1 1x1 $name
   # [[ x$ptopo == x ]] && \
   #    runonecase cfg_$name 1x4x1 1x1 $name

   name=HRDPS-E
   runonecase cfg_$name 4x2x1 1x1 $name
   # [[ x$ptopo == x ]] && \
   #    runonecase cfg_$name 1x4x1 1x1 $name

   name=HRDPS-E-2.5
   runonecase cfg_$name 4x2x1 1x1 $name
   # [[ x$ptopo == x ]] && \
   #    runonecase cfg_$name 1x4x1 1x1 $name

   #name=HRDPS-W
   #runonecase cfg_$name 4x1x1 1x1 $name
   # [[ x$ptopo == x ]] && \
   #    runonecase cfg_$name 1x4x1 1x1 $name

   # name=HRDPS-W-2.5
   # runonecase cfg_$name 4x1x1 1x1 $name
   # [[ x$ptopo == x ]] && \
   #    runonecase cfg_$name 1x4x1 1x1 $name
fi

if [[ x$mycase == xtestcases16 ]] ; then
   name=RI
   runonecase cfg_$name 4x2x1 1x1 $name

   name=RDPS
   runonecase cfg_$name 4x4x1 1x1 $name

   name=HRDPS-E
   runonecase cfg_$name 4x4x1 1x1 $name

   name=HRDPS-E-2.5
   runonecase cfg_$name 4x4x1 1x1 $name
fi



if [[ x$mycase == xall ]] ; then
   for item in cfg_* ; do
      name=${item##cfg_}
      runonecase $item 1x1x1 1x1 ${name:-RI}
   done
fi


# if [[ x$mycase == xtestcases ]] ; then

#    topo1="2x1"
#    topo2="1x2"
#    item=RI
#    runonecase GEM_cfgs ${topo1}x1 1x1 $item
#    runonecase GEM_cfgs ${topo2}x1 1x1 $item
#    ok="$($mydiff blocstat_${item}_p${topo1}x1_b1x1_${BASE_ARCH}.log blocstat_${item}_p${topo1}x1_b1x1_${BASE_ARCH}.log)"
#    if [[ x"$ok" == x ]] ; then
#       if [[ -s blocstat_${item}_p${topo2}x1_b${topo1}_${BASE_ARCH}.log ]] ; then
#          echo "OK $item MPI bloc repro ${topo2} ${topo1}"
#       else
#          echo "FAIL $item MPI bloc repro ${topo2} ${topo1} (cannot compare, run failed)"
#       fi
#    fi



#    for item in pilot_glb pilot_lam2 test_yy yinyang; do
#       topo1="2x1"
#       topo2="1x2"
#       runonecase cfg_$item ${topo1}x1 ${topo1} $item
#       ok="$($mydiff 
# blocstat_${item}_p${topo2}x1_b${topo1}_${BASE_ARCH}.log 
# blocstat_${item}_p${topo2}x1_b${topo2}_${BASE_ARCH}.log)"
#       else
#          echo "FAIL $item MPI bloc repro ${topo2} ${topo1}"
#       fi
#       ok="$($mydiff blocstat_${item}_p${topo1}x1_b${topo1}_${BASE_ARCH}.log blocstat_${item}_p${topo2}x1_b${topo2}_${BASE_ARCH}.log)"
#       if [[ x"$ok" == x ]] ; then
#          if [[ -s blocstat_${item}_p${topo1}x1_b${topo1}_${BASE_ARCH}.log ]] ; then
#             echo "OK $item MPI topo repro ${topo2} ${topo1}"
#          else
#             echo "FAIL $item MPI topo repro ${topo2} ${topo1} (cannot compare, run failed)"
#          fi
#       else
#          echo "FAIL $item MPI topo repro ${topo2} ${topo1}"
#       fi
#    done
#    exit 0
# fi

# if [[ x$mycase == xmpirepro ]] ; then
#    [[ x$mycase != xall ]] && runonecase cfg_pilot_glb 2x1x1 1x1 pilot_glb
#    runonecase cfg_pilot_glb 1x1x1 1x1 pilot_glb
#    ok="$($mydiff blocstat_pilot_glb_p1x1x1_b*_${BASE_ARCH}.log blocstat_pilot_glb_p[2-9]x[1-9]x1_b*_${BASE_ARCH}.log)"
#    if [[ x"$ok" == x ]] ; then
#       echo "OK pilot_glb MPI repro 2x1x1 1x1x1"
#    else
#       echo "FAIL pilot_glb MPI repro 2x1x1 1x1x1"
#    fi
#    [[ x$mycase != xall ]] && runonecase cfg_pilot_lam2 2x1x1 2x1 pilot_lam2
#    runonecase cfg_pilot_lam2 1x1x1 1x1 pilot_lam2
#    ok="$($mydiff blocstat_pilot_lam2_p1x1x1_b*_${BASE_ARCH}.log blocstat_pilot_lam2_p[2-9]x[1-9]x1_b*_${BASE_ARCH}.log)"
#    if [[ x"$ok" == x ]] ; then
#       echo "OK pilot_lam2 MPI repro 2x1x1 1x1x1"
#    else
#       echo "FAIL pilot_lam2 MPI repro 2x1x1 1x1x1 (Traj crop may cause this)"
#    fi
#    [[ x$mycase == xmpirepro ]] && exit 0
# fi

# if [[ x$mycase == xomprepro ]] ; then
#    runonecase cfg_pilot_glb 1x1x1 1x1 pilot_glb
#    runonecase cfg_pilot_glb 1x1x4 1x1 pilot_glb
#    ok="$($mydiff blocstat_pilot_glb_p1x1x1_b*_${BASE_ARCH}.log blocstat_pilot_glb_p1x1x[2-9]_b*_${BASE_ARCH}.log)"
#    if [[ x"$ok" == x ]] ; then
#       echo "OK pilot_glb OMP repro 1x1x1 1x1x4"
#    else
#       echo "FAIL pilot_glb OMP repro 1x1x1 1x1x4"
#    fi
#    runonecase cfg_pilot_lam2 1x1x1 1x1 pilot_lam2
#    runonecase cfg_pilot_lam2 1x1x4 1x1 pilot_lam2
#    ok="$($mydiff blocstat_pilot_lam2_p1x1x1_b*_${BASE_ARCH}.log blocstat_pilot_lam2_p1x1x[2-9]_b*_${BASE_ARCH}.log)"
#    if [[ x"$ok" == x ]] ; then
#       echo "OK pilot_lam2 OMP repro 1x1x1 1x1x4"
#    else
#       echo "FAIL pilot_lam2 OMP repro 1x1x1 1x1x4"
#    fi
#    [[ x$mycase == xomprepro ]] && exit 0
# fi


if [[ x$hasranonecase == x0 ]] ; then
   name=${mycase}
   runonecase cfg_$mycase 1x1x1 1x1 ${name:-RI}
fi
