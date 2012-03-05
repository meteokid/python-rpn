#!/bin/ksh

myself=$0
myecho="echo -e"
[[ "x$(echo -e)" != "x" ]] && myecho="echo"

DESC='Run GEM test cases interactively'
USAGE="USAGE: ${myself##*/} [-h] [--nompi | --ptopo NPEXxNPEY] [--make] [--stdout] [-c CASENAME | -l]"

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
    --make      : build GEM abs (defautl is not to make)
    --stdout    : send model output to stdout instead of log file
    -c CASENAME : run model with config cfg_CASENAME (default is all)
    -l          : list avail cfg
EOF
}

runonecase() {
	 _dircfg=$1
	 _ptopo=$2
	 _name=$3
	 _isnest=${4:-no}
	 
	 if [[ x$nompi != x ]] ; then
		  _ptopo=1x1
	 elif [[ x$ptopo != x ]] ; then
		  _ptopo=$ptopo
	 fi
	 
	 if [[ x${_isnest} == xno ]] ; then 
		  rm -rf RUNENT/*
		  rm -rf RUNMOD/*
		  cmd="Um_runent.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1"
		  if [[ x$stdout == x0 ]] ; then
				$cmd >runent_${_name}_${BASE_ARCH}.log 2>&1
				echo "runent $(grep _status runent_${_name}_${BASE_ARCH}.log)"
		  else
				$cmd
		  fi
	 fi
	 cmd="Um_runmod.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1 -out_block ${_ptopo} $nompi"
	 if [[ x$stdout == x0 ]] ; then
		  $cmd >runmod_${_name}_${BASE_ARCH}.log 2>&1
		  echo "runmod $(grep _status runmod_${_name}_${BASE_ARCH}.log)"
	 else
		  $cmd
	 fi
}

savepilotoutput() {
	 _here=$(pwd)
	 cd RUNMOD_output
	 rm -rf input
	 mkdir input
	 find . -name '3df*' -exec mv {} input \;
	 cd ${_here}
}

#---- Inline Options -------------------------------------------------

domake=0
mycase=all
stdout=0
nompi=''
ptopo=''
dolist=0
previous=""
while [[ $# -gt 0 ]] ; do
    case $1 in
        (-h|--help) usage_long; exit 0;;
		  (--nompi) nompi='-nompi';;
		  (--make) domake=1;;
		  (--ptopo) [[ $# -gt 1 ]] && ptopo=$2 && shift ;;
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

myversion=4.4.0-b3

if [[ x$ATM_MODEL_VERSION != x${myversion} ]] ; then
	 cat <<EOF
ERROR: GEM version $myversion requested (${ATM_MODEL_VERSION:-NONE} loaded)
---- ABORT ----
EOF
	 exit 1
	 #. s.ssmuse.dot GEM/tests/${myversion}
fi

if [[ ! -r .exper_cour ]] ; then
	 ouv_exp base -RCSPATH $gemdyn/RCS
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

set -x

if [[ x$mycase == xall ]] ; then

runonecase cfg_pilot_glb 2x1 pilot_glb
savepilotoutput
runonecase cfg_nest 2x2 nest_glb isnest

runonecase cfg_pilot_lam 2x1 pilot_lam
savepilotoutput
runonecase cfg_nest 2x2 nest_lam isnest

runonecase cfg_pilot_glb_digf 2x1 pilot_glb_digf
runonecase cfg_pilot_glb_digftr 2x1 pilot_glb_digftr

runonecase cfg_pilot_lam 2x1 pilot_lam_topoflt

elif [[ x$mycase == xnest ]] ; then

runonecase cfg_pilot_glb 2x1 pilot_glb
savepilotoutput
runonecase cfg_nest 2x2 nest_glb isnest

runonecase cfg_pilot_lam 2x1 pilot_lam
savepilotoutput
runonecase cfg_nest 2x2 nest_lam isnest

elif [[ x$mycase == xdigf ]] ; then

runonecase cfg_pilot_glb_digf 2x1 pilot_glb_digf
runonecase cfg_pilot_glb_digftr 2x1 pilot_glb_digftr

elif [[ x$mycase == xlam || x$mycase == xglb ]] ; then

runonecase cfg_pilot_$mycase 2x1 pilot_$mycase
savepilotoutput
runonecase cfg_nest 2x2 nest_$mycase isnest

if [[ x$mycase == xlam ]] ; then
runonecase cfg_pilot_lam 2x1 pilot_lam_topoflt
fi

elif [[ x$mycase == xpilot ]] ; then

runonecase cfg_pilot_glb 2x1 pilot_glb
runonecase cfg_pilot_lam 2x1 pilot_lam
runonecase cfg_pilot_glb_digf 2x1 pilot_glb_digf
runonecase cfg_pilot_glb_digftr 2x1 pilot_glb_digftr
runonecase cfg_pilot_lam 2x1 pilot_lam_topoflt

else

runonecase cfg_$mycase 2x1 $mycase

fi
