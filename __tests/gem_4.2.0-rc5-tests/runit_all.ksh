#!/bin/ksh

myversion=4.2.0-rc5

echo "Make sure path are updated in configexp.cfg of nest cases." 1>&2

if [[ x$ATM_MODEL_VERSION != x${myversion} ]] ; then
#	 . s.ssmuse.dot GEM/tests/${myversion}
	 echo "ERROR: GEM version ${myversion} is expected, $ATM_MODEL_VERSION loaded" 1>&2
	 echo '---- ABORT ----' 1>&2
	 exit 1
fi

if [[ ! -r .exper_cour ]] ; then
	 ouv_exp base -RCSPATH $gemdyn/RCS
fi

linkit

mycase=all
if [[ x$1 == x--make ]] ; then
	 r.make_exp
	 make gem
elif [[ x$1 == x-c ]] ; then
	 mycase="$2"
fi

runonecase() {
	 _dircfg=$1
	 _ptopo=$2
	 _name=$3
	 _isnest=${4:-no}
	 

	 if [[ x${_isnest} == xno ]] ; then 
		  rm -rf RUNENT/*
		  rm -rf RUNMOD/*
		  Um_runent.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1 >runent_${_name}_${BASE_ARCH}.log 2>&1
		  echo "runent $(grep _status runent_${_name}_${BASE_ARCH}.log)"
	 fi
	 Um_runmod.ksh -cfg 1:1 -dircfg ${_dircfg} -ptopo ${_ptopo}x1 -out_block ${_ptopo} >runmod_${_name}_${BASE_ARCH}.log 2>&1
	 echo "runmod $(grep _status runmod_${_name}_${BASE_ARCH}.log)"
}

savepilotoutput() {
	 _here=$(pwd)
	 cd RUNMOD_output
	 rm -rf input
	 mkdir input
	 find . -name '3df*' -exec mv {} input \;
	 cd ${_here}
}

set -x

if [[ x$mycase == xall ]] ; then

runonecase cfg_pilot_glb 2x1 pilot_glb
savepilotoutput
runonecase cfg_nest 2x2 nest_glb isnest

runonecase cfg_pilot_lam 2x1 pilot_lam
savepilotoutput
runonecase cfg_nest 2x2 nest_lam isnest


elif [[ x$mycase == xlam || x$mycase == xglb ]] ; then

runonecase cfg_pilot_$mycase 2x1 pilot_$mycase
savepilotoutput
runonecase cfg_nest 2x2 nest_$mycase isnest

elif [[ x$mycase == xpilot ]] ; then

runonecase cfg_pilot_glb 2x1 pilot_glb
runonecase cfg_pilot_lam 2x1 pilot_lam

else

runonecase cfg_$mycase 2x1 $mycase

fi
