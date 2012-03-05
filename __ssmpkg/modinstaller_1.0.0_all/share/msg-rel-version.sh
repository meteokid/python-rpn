#!/bin/ksh

__basedir=__BASEDIR__
#__bundlename=${ATM_MODEL_BNDL:-__BUNDLENAME__}
#__softname=${ATM_MODEL_NAME:-GEMDM}
__bundlename=__BUNDLENAME__
__softname=GEMDM

here=`pwd`
cd ${__basedir} 2>/dev/null
#__latestver=$(ls -t ${__bundlename%.*}.* 2>/dev/null)
#__latestver=$(echo ${__latestver})
#__latestver=${__latestver%% *}
#__latestver=${__latestver%.*}
__latestverlist="$(ls -rt ${__bundlename%.*}.* 2>/dev/null)"
__latestver=$(ls ${__bundlename}.* 2>/dev/null)
for item in ${__latestverlist} ; do
	 [[ -r ${__basedir}/$item ]] && __latestver=$item
done
__latestver=${__latestver%.*}

cd $here

[[ x$__latestver == x ]] && __latestver=${__bundlename}

mylang=$(echo ${CMCLNG} | cut -c1-2)

echo_newver() {
	 if [[ x${mylang:-en} != xfr ]] ; then
		  cat <<EOF
============================================================================
NOTE   : A newer version of ${__softname} with some bugfix is available
         Please use the latest version as this one (${ATM_MODEL_BNDL:-$__bundlename})
         may contains bugs that are now fixed.
         . s.ssmuse.dot ${__latestver}
============================================================================
EOF
	 else
		  cat <<EOF
============================================================================
NOTE     : Une nouvelle version de ${__softname} avec des correctifs
           est disponible.
           SVP utiliser la derniere version puisque celle-ci (${ATM_MODEL_BNDL:-$__bundlename})
           peut contenir des defectuosites maintenant corrigees.
           . s.ssmuse.dot ${__latestver}
============================================================================
EOF
	 fi
}


if [[ x${__bundlename} != x${__latestver} ]] ; then
	 echo_newver
fi
