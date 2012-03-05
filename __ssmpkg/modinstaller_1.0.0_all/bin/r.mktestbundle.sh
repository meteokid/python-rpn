#!/bin/ksh

basedir=$(true_path ~/SsmBundles)

DESC='Create a test bundle from GEM from another one'
USAGE="USAGE: ${0##*/} FROM_BNDL TO_BNDL "

usage_long() {
	 toto=$(echo -e $USAGE)
	 more <<EOF
$toto
EOF
}

from_bndl="${1}.bndl"
to_bndl="${2}.bndl"

if [[ x$1 == x || x$2 == x ]] ; then
	 echo "ERROR: Wrong number of args" 1>&2
	 echo -e $USAGE 1>&2
	 echo "---- ABORT ----"  1>&2
	 exit 1
fi

if [[ ! -r $(true_path $from_bndl 2>/dev/null) ]] ; then
	 from_bndl="$basedir/$from_bndl"
fi

if [[ ! -r $(true_path $from_bndl) ]] ; then
	 echo "ERROR: FROM_BNDL ($from_bndl) not existing or readable" 1>&2
	 echo -e $USAGE 1>&2
	 echo "---- ABORT ----"  1>&2
	 exit 1
fi

if [[ x$(echo $to_bndl|cut -c1) != x/ ]] ; then
	 to_bndl="$basedir/$to_bndl"
fi

cat $from_bndl | sed 's|GEM/others/gem-home-path|GEM/others/gem-home-path GEM/others/renametotest|' > $to_bndl
chmod 555 $to_bndl

ls -l $to_bndl