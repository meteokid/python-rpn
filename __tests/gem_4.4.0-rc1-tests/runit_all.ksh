#!/bin/ksh

myself=$0
myecho="echo -e"
[[ "x$(echo -e)" != "x" ]] && myecho="echo"

DESC='Run GEM test cases interactively'
USAGE="USAGE: ${myself##*/} [-h] [--nompi | [--ptopo NPEXxNPEY] [--btopo NPEXxNPEY]] [--make] [--stdout] [-c CASENAME | -l]"

fromversion=4.4.0-b14
myversion=4.4.0-rc1
#TODO: on AIX* test openmpi running and reproducibility of 2 different omp

#export PATH=${ATM_MODEL_BINDIR}:$PATH

#---- Functions ------------------------------------------------------
#====
usage_long() {
	 toto=$($myecho $USAGE)
	 more <<EOF
$DESC

$toto

Options:
    -h, --help  : print this help
    --nompi     : run w/o mpi (must be compile w/ nompi option
    --ptopo 9x9 : force proc topology to NPEXxNPEY
    --btopo 9x9 : force i/o topology to NPEXxNPEY
    --make      : build GEM abs (defautl is not to make)
    --stdout    : send model output to stdout instead of log file
    -c CASENAME : run model with config cfg_CASENAME (default is all)
    -l          : list avail cfg
EOF
}

myupcfg() {
	 _dircfg=${1:-/dev/null}
	 _here=$(pwd)
	 cd ${_dircfg}/cfg_0001
	 gem_upcfg --from ${fromversion} --to ${myversion}
	 cd ${_here}
}


runonecase() {
	 _dircfg=$1
	 _ptopo=$2
	 _btopo=$3
	 _name=$4
	 _runpil=${5:-dorunpil}

	 hasranonecase=1

	 if [[ x$stdout == x0 ]] ; then
		  myupcfg ${_dircfg} >/dev/null
	 else
		  myupcfg ${_dircfg}
	 fi
	 
	 if [[ x$nompi != x ]] ; then
		  _ptopo=1x1
		  _btopo=1x1
	 else
		  if [[ x$ptopo != x ]] ; then
				_ptopo=$ptopo
		  fi
		  _btopo=$_ptopo
		  if [[ x$btopo != x ]] ; then
				_btopo=$btopo
		  fi
	 fi
	 
	 _theoc=''
	 if [[  x${_runpil} == xtheoc ]] ; then
		  _theoc='-theoc'
	 fi

	 echo '+ runonecase' $_dircfg $_ptopo $_btopo $_name $_runpil

	 if [[ x${_runpil} == xdorunpil ]] ; then 
		  rm -rf RUNENT/*
		  rm -rf RUNMOD/*
		  cmd="Um_runent.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1"
		  if [[ x$stdout == x0 ]] ; then
				logfile=runent_${_name}_p${_ptopo}x1_b${_btopo}_${BASE_ARCH}.log
				$cmd > $logfile 2>&1
				echo "runent $(grep _status $logfile)"
		  else
				$cmd
		  fi
	 fi
	 cmd="Um_runmod.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1 -out_block ${_btopo} $nompi ${_theoc}"
	 if [[ x$stdout == x0 ]] ; then
		  logfile=runmod_${_name}_p${_ptopo}x1_b${_btopo}_${BASE_ARCH}.log
		  $cmd >$logfile 2>&1
		  echo "runmod $(grep _status $logfile)"
	 else
		  $cmd
	 fi
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
	 mv ${_dircfg}/cfg_0001/configexp.cfg ${_dircfg}/cfg_0001/configexp.cfg-0
	 cat ${_dircfg}/cfg_0001/configexp.cfg-0 | grep -v UM_EXEC_model_input \
		  > ${_dircfg}/cfg_0001/configexp.cfg
	 cat >> ${_dircfg}/cfg_0001/configexp.cfg <<EOF
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
while [[ $# -gt 0 ]] ; do
    case $1 in
        (-h|--help) usage_long; exit 0;;
		  (--nompi) nompi='-nompi';;
		  (--make) domake=1;;
		  (--ptopo) [[ $# -gt 1 ]] && ptopo=$2 && shift ;;
		  (--btopo) [[ $# -gt 1 ]] && btopo=$2 && shift ;;
		  (-l|--list) dolist=1;;
		  (--stdout) stdout=1;;
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
	 echo 'lam   #(all lam configs)'
	 echo 'glb   #(all glb configs)'
	 echo 'pilot #(all pilot configs)'
	 exit 0
fi

if [[ x$ATM_MODEL_VERSION != x${myversion} ]] ; then
	 cat <<EOF
ERROR: GEM version $myversion requested (${ATM_MODEL_VERSION:-NONE} loaded)
---- ABORT ----
EOF
	 exit 1
	 #. s.ssmuse.dot GEM/tests/${myversion}
fi

linkit

if [[ x$domake == x1 ]] ; then
   if [[ ! -r .exper_cour ]] ; then
	 ouv_exp base -RCSPATH $modelutils/RCS $gemdyn/RCS
   fi
	 r.make_exp
	 if [[ x$nompi != x ]] ; then
		  make gem_nompi
	 else
		  make gem
	 fi
#else
   #s.use maingemntr_REL_${BASE_ARCH}.Abs as maingemntr_${BASE_ARCH}.Abs
   #s.use maingemdm_REL_${BASE_ARCH}.Abs as maingemdm_${BASE_ARCH}.Abs
#   rm -f maingem*_${BASE_ARCH}.Abs
#   ln -s $(which maingemntr_REL_${BASE_ARCH}.Abs) maingemntr_${BASE_ARCH}.Abs
#   ln -s $(which maingemdm_REL_${BASE_ARCH}.Abs) maingemdm_${BASE_ARCH}.Abs
fi


hasranonecase=0

if [[ x$mycase == xnest || x$mycase == xpilot || x$mycase == xall \
	 || x$mycase == xlam || x$mycase == xglb ]] ; then

	 if [[ x$mycase != xlam ]] ; then
		  runonecase cfg_pilot_glb 2x1 1x1 pilot_glb
		  if [[ x$mycase == xnest || x$mycase == xall ]] ; then
				savepilotoutput cfg_nest2
				runonecase cfg_nest2 2x2 1x1 nest2_glb noPil
		  fi
	 fi

	 if [[ x$mycase != xglb ]] ; then
		  runonecase cfg_pilot_lam-mid 2x1 1x1 pilot_lam-mid
		  runonecase cfg_pilot_lam2 2x1 1x1 pilot_lam2
		  if [[ x$mycase == xnest || x$mycase == xall ]] ; then
				savepilotoutput cfg_nest2
				runonecase cfg_nest2 2x2 1x1 nest2_lam2 noPil
		  fi
	 fi

	 if [[ x$mycase != xnest ]] ; then
		  if [[ x$mycase != xglb ]] ; then
				runonecase cfg_pilot_lam2_topoflt 2x1 1x1 pilot_lam2_topoflt
		  fi
		  if [[ x$mycase != xlam ]] ; then
				runonecase cfg_pilot_glb_digf 2x1 1x1 pilot_glb_digf
				runonecase cfg_pilot_glb_digftr 2x1 1x1 pilot_glb_digftr
				runonecase cfg_midglb 2x1 1x1 midglb
				runonecase cfg_glb_cccmarad 2x1 1x1 glb_cccmarad
				runonecase cfg_glb_andre 2x1 1x1 glb_andre
            runonecase cfg_yinyang 2x1 1x1 yinyang
            #TODO: activate the following tests
            #runonecase cfg_glb_icelac 2x1 1x1 glb_icelac
		  fi
	 fi

	 if [[ x$ptopo == x && x$btopo == x ]] ; then
		  runonecase cfg_pilot_glb 2x1 2x1 pilot_glb
		  runonecase cfg_pilot_glb 1x1 1x1 pilot_glb
		  runonecase cfg_pilot_lam2 2x1 2x1 pilot_lam2
		  runonecase cfg_pilot_lam2 1x1 1x1 pilot_lam2
	 fi

fi

if [[ x$mycase == xall || x$mycase == xtheo ]] ; then
	 runonecase cfg_theo-mtn 8x1 1x1 theo-mtn theoc
fi

if [[ x$mycase == xdigf ]] ; then
	 runonecase cfg_pilot_glb_digf 2x1 1x1 pilot_glb_digf
	 runonecase cfg_pilot_glb_digftr 2x1 1x1 pilot_glb_digftr
fi


if [[ x$hasranonecase == x0 ]] ; then
	 runonecase cfg_$mycase 2x1 1x1 $mycase
fi
