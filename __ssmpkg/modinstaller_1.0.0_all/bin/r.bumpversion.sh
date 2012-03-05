#!/bin/ksh

DESC='Increase an ssm pkg version number'
USAGE="USAGE: ${0##*/} [-h] \\ \n
        \t PACKAGENAME_VERSION \\ \n 
        \t [--rc || --beta || --alpha || --bugfix ]\\ \n
        \t [--minor || --major ]
        \t [-b branch_name] "

usage_long() {
	 toto=$(echo -e $USAGE)
	 more <<EOF
$toto
EOF
}

myecho=echo

isminor=0
ismajor=0
isbugfix=0
branchname=''
posargs=""
posargsnum=0
while [[ $# -gt 0 ]] ; do
    case $1 in
        (-h|--help) echo $DESC ; usage_long; exit 0;;
		  (--bugfix) isbugfix=1 ;;
		  (--rc)     isbugfix=2 ;;
		  (--beta)   isbugfix=3 ;;
		  (--alpha)  isbugfix=4 ;;
		  (--minor)  isminor=1 ;;
		  (--major)  ismajor=1 ;;
		  (-b) branchname=$2; shift ;;
        -*) echo "ERROR: Unrecognized option $1"
            echo -e $USAGE
            echo "---- ABORT ----"
            exit 1;;
        *) posargs="$posargs $1" ; ((posargsnum=posargsnum+1));;
    esac
    shift
done

if [[ posargsnum -ne 1 ]] ; then
	 echo "ERROR: Wrong number of args, need to provide PACKAGENAME_VERSION" 1>&2
	 echo -e $USAGE 1>&2
	 echo "---- ABORT ----"  1>&2
	 exit 1
fi

#set -x 

pkgname=${posargs%_*}
pkgname=${pkgname# *}
pkgversion=${posargs#*_} #M.m[-b][.[abrc]0-9]

pkgversion1=${pkgversion%%.*}
pkgversion23=${pkgversion#*.}
pkgversion2=${pkgversion23%.*}
pkgversion3=${pkgversion23#*.}

hasbranch=${pkgversion2#*-}
[[ x$hasbranch == x${pkgversion2} ]] && hasbranch=""
[[ x$hasbranch != x ]] && pkgversion2=${pkgversion2%%-*}

hasbugfix=1
hasabrc=$(echo $pkgversion3 | cut -c1)
if [[ x$hasabrc == xr ]] ; then
	 hasbugfix=2
	 pkgversion3=$(echo $pkgversion3 | cut -c3-)
elif [[ x$hasabrc == xb ]]  ; then
	 hasbugfix=3
	 pkgversion3=$(echo $pkgversion3 | cut -c2-)
elif [[ x$hasabrc == xa  ]] ; then
	 hasbugfix=4
	 pkgversion3=$(echo $pkgversion3 | cut -c2-)
fi

pkgversion1b=$pkgversion1
pkgversion2b=$pkgversion2
pkgversion3b=$pkgversion3
pkgbranch2=$hasbranch
pkgabrc=''

if [[ ismajor -eq 1 ]] ; then
	 ((pkgversion1b=pkgversion1b+1))
	 pkgversion2b=0
	 pkgbranch2=''
	 pkgversion3b=0
	 [[ isbugfix -eq 2 ]] && pkgabr='rc'
	 [[ isbugfix -eq 3 ]] && pkgabr='b'
	 [[ isbugfix -eq 4 ]] && pkgabr='a'
elif [[ isminor -eq 1 ]] ; then
	 ((pkgversion2b=pkgversion2b+1))
	 pkgbranch2=''
	 pkgversion3b=0
	 [[ isbugfix -eq 2 ]] && pkgabr='rc'
	 [[ isbugfix -eq 3 ]] && pkgabr='b'
	 [[ isbugfix -eq 4 ]] && pkgabr='a'
elif [[ x$branchname != x && x$branchname != x$hasbranch ]] ; then
	 pkgbranch2=$branchname
	 pkgversion3b=0
	 [[ isbugfix -eq 2 ]] && pkgabr='rc'
	 [[ isbugfix -eq 3 ]] && pkgabr='b'
	 [[ isbugfix -eq 4 ]] && pkgabr='a'
else
	 if [[ isbugfix -eq 0 || isbugfix -eq hasbugfix ]] ; then
		  ((pkgversion3b=pkgversion3b+1))
		  [[ hasbugfix -eq 2 ]] && pkgabr='rc'
		  [[ hasbugfix -eq 3 ]] && pkgabr='b'
		  [[ hasbugfix -eq 4 ]] && pkgabr='a'
	 elif [[ isbugfix -lt hasbugfix ]] ; then
		  ((pkgversion3b=0))
		  [[ isbugfix -eq 2 ]] && pkgabr='rc'
		  [[ isbugfix -eq 3 ]] && pkgabr='b'
		  [[ isbugfix -eq 4 ]] && pkgabr='a'
	 elif [[ isbugfix -gt hasbugfix ]] ; then
		  echo "ERROR: Trying to downgrade a version... set --major or --minor or -b" 1>&2
		  echo -e $USAGE 1>&2
		  echo "---- ABORT ----" 1>&2
		  exit 1
	 fi
fi

[[ x$pkgbranch2 != x ]] && pkgbranch2="-$pkgbranch2"

pkgversion2="${pkgversion1b}.${pkgversion2b}${pkgbranch2}.${pkgabr}${pkgversion3b}"
echo "Bumping $pkgname from ${pkgversion} to ${pkgversion2}"

for item in ${pkgname}_${pkgversion}_* ; do
	 myarch=${item##*_}
	 rsync -a $item/ ${pkgname}_${pkgversion2}_$myarch
	 for item2 in maint/include maint/make-ssm .ssm.d/control ; do
		  chmod -R u+w ${pkgname}_${pkgversion2}_${myarch}
		  cat $item/$item2 | sed "s:${pkgversion}:${pkgversion2}:gi" > ./${pkgname}_${pkgversion2}_${myarch}/${item2}
	 done	 
done

exit 0
