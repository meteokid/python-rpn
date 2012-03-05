#!/bin/ksh

__basedir0=/home/ordenv/ssm-domains/ssm-setup-1.0/dot-profile-setup_1.0_multi/notshared/data/ssm_domains
__basedir=__BASEDIR__
#__bundlename=${ATM_MODEL_BNDL:-__BUNDLENAME__}
#__softname=${ATM_MODEL_NAME:-GEMDM}
__bundlename=__BUNDLENAME__
__softname=GEMDM
myls=$(which ls)

__logfile="~armnenv/gem-usage.log"

#==========================================================================
log_usage() {
	 echo "$(date '+%F.%H%M%S'):${__bundlename}:$(whoami)@$(hostname):(${TRUE_HOST};${BASE_ARCH})" >> ~armnenv/gem-usage.log 2>/dev/null
}

is_bndl_ok() {
	 __mybndlname=$1
	 __isok=1
	 if [[ -r ${__basedir}/${__mybndlname} ]] ; then
		  for item2 in $(cat ${__basedir}/${__mybndlname} 2>/dev/null) ; do
				myarch=${item2%:*}
				mydom=${item2#*:}
				if [[ x$myarch == x$BASE_ARCH || x$myarch == x$mydom ]] ; then
					 __found=0
					 for __mydir0 in ${__basedir} ${__basedir0} ; do
						  [[ -r ${__mydir0}/$mydom ]] && __found=1
						  if [[ ${__found} == 0 ]] ; then
								for __ext in sh bndl ; do
									 [[ -r ${__mydir0}/${mydom}.${__ext} ]] && __found=1
								done
						  fi
					 done
					 [[ ${__found} == 0 ]] && __isok=0
				fi
		  done
	 else
		  __isok=0
	 fi
	 echo $__isok
}

echo_expver() {
	 if [[ x${mylang:-en} != xfr ]] ; then
		  cat <<EOF
WARNING: You are using an experimental version of ${__softname} (${ATM_MODEL_BNDL:-$__bundlename}).
         It will only be available for a limited time (until next version).
         Please report bugs to the developer team.
EOF
	 else
		  cat <<EOF
ATTENTION: Vous utilisez une version experimentale de ${__softname} (${ATM_MODEL_BNDL:-$__bundlename}).
           Elle ne sera offerte que pour une periode limite
           (jusqu'a la sortie de la prochaine version).
           SVP informer les developpeurs des bugs rencontres.
EOF
	 fi
}


echo_newver() {
	 if [[ x${mylang:-en} != xfr ]] ; then
		  cat <<EOF
NOTE   : A newer version of ${__softname} is available.
         Please use the latest version as this one (${ATM_MODEL_BNDL:-$__bundlename})
         will be retired soon.
         . s.ssmuse.dot ${__latestver}

EOF
	 else
		  cat <<EOF
NOTE     : Une nouvelle version de ${__softname} est disponible.
           SVP utiliser la derniere version puisque celle-ci (${ATM_MODEL_BNDL:-$__bundlename})
           sera innaccessible sous peu.
           . s.ssmuse.dot ${__latestver}

EOF
	 fi
	 echo_expver
}

echo_delver() {
	 if [[ x${mylang:-en} != xfr ]] ; then
		  cat <<EOF
ERROR: this experimental version of ${__softname} (${__bundlename}) 
       is no longer available

       Please use the latest available version
       . s.ssmuse.dot ${__latestver}
EOF
	 else
		  cat <<EOF
ERREUR: cette version experimental de ${__softname} (${__bundlename}) 
        n'est plus disponible.

        SVP utiliser la derniere version disponible
        . s.ssmuse.dot ${__latestver}
EOF
	 fi
}

#==========================================================================

log_usage

here=`pwd`
cd ${__basedir} 2>/dev/null
__latestverlist="$($myls -rt1 ${__bundlename%.*}.*.bndl 2>/dev/null)"
__latestver=$($myls ${__bundlename}.* 2>/dev/null)
__okver=$(is_bndl_ok ${__bundlename}.bndl)
for item in ${__latestverlist} ; do
	 __isok=$(is_bndl_ok ${item})
    [[ x${__isok} == x1 && ! -L $item ]] && __latestver=$item
done
__latestver=${__latestver%.*}

cd $here

[[ x$__latestver == x ]] && __latestver=${__bundlename}

mylang=$(echo ${CMCLNG} | cut -c1-2)

cat <<EOF
============================================================================
EOF

if [[ __okver -eq 1 ]] ; then
	 if [[ x${__bundlename} == x${__latestver} ]] ; then
		  echo_expver
	 else
		  echo_newver
	 fi
else
	 echo_delver
fi

cat <<EOF
============================================================================
EOF
