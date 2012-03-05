#!/bin/ksh

myself=$0
myecho="echo -e"
[[ "x$(echo -e)" != "x" ]] && myecho="echo"

DESC='Run GEM test cases interactively'
USAGE="USAGE: ${myself##*/} [-h] [--nompi | [--ptopo NPEXxNPEY] [--btopo NPEXxNPEY]] [--make] [--stdout] [-c CASENAME | -l]"

myversion=4.4.0-b10

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

runonecase() {
	 _dircfg=$1
	 _ptopo=$2
	 _btopo=$3
	 _name=$4
	 _isnest=${5:-no}

	 hasranonecase=1
	 
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
	 
	 echo '+ runonecase' $_dircfg $_ptopo $_btopo $_name $_isnest

	 if [[ x${_isnest} == xno ]] ; then 
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
	 cmd="Um_runmod.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1 -out_block ${_btopo} $nompi"
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

if [[ ! -r .exper_cour ]] ; then
	 ouv_exp base -RCSPATH $modelutils/RCS $gemdyn/RCS
fi

linkit

if [[ x$domake == x1 ]] ; then
	 r.make_exp
	 if [[ x$nompi != x ]] ; then
		  make gem_nompi
	 else
		  make gem
	 fi
fi


hasranonecase=0

if [[ x$mycase == xnest || x$mycase == xpilot || x$mycase == xall \
	 || x$mycase == xlam || x$mycase == xglb ]] ; then

	 
	 if [[ x$mycase != xlam ]] ; then
		  runonecase cfg_pilot_glb 2x1 1x1 pilot_glb
		  if [[ x$mycase == xnest || x$mycase == xall ]] ; then
				savepilotoutput cfg_nest2
				runonecase cfg_nest2 2x2 1x1 nest2_glb isnest
		  fi
	 fi

	 if [[ x$mycase != xglb ]] ; then
		  runonecase cfg_pilot_lam2 2x1 1x1 pilot_lam2
		  if [[ x$mycase == xnest || x$mycase == xall ]] ; then
				savepilotoutput cfg_nest2
				runonecase cfg_nest2 2x2 1x1 nest2_lam2 isnest
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

fi

if [[ x$mycase == xdigf ]] ; then
	 runonecase cfg_pilot_glb_digf 2x1 1x1 pilot_glb_digf
	 runonecase cfg_pilot_glb_digftr 2x1 1x1 pilot_glb_digftr
fi


if [[ x$hasranonecase == x0 ]] ; then
	 runonecase cfg_$mycase 2x1 1x1 $mycase
fi
