#!/bin/ksh

basedir=$(true_path ~/SsmBundles)
SSMVERSION=$(ssm version | cut -d. -f1)

if [[ $SSMVERSION -lt 10 ]] ; then
   echo 'ERROR: Need to use SSM version >= 10' ", using: $(ssm version)"
   exit 1
fi

DESC='Create a GEM bundle with all GEM ssm pkg published in a single ssm domain'
USAGE="USAGE: ${0##*/} [-v] FROM_BNDL TO_BNDL"

usage_long() {
    toto=$(echo -e $USAGE)
    more <<EOF
$toto
EOF
}

myecho=""
[[ x"$1" == x"-n" ]] && myecho=echo && shift

gemversion=${1##*/}

bndlname=${1}
bndlname2=${2}

from_bndl="${1}.bndl"
to_bndl="${2}.bndl"

name0=$(echo ${bndlname%%/*} | tr 'A-Z' 'a-z')
remains=${bndlname#*/}
namex=${remains%%/*}/
remains=${remains#*/}
namev=${remains#*/}

[[ xy$namex == xy${namev}/ ]] && namex=
[[ xy$namex == xy${name0}/ ]] && namex=

Prefixes=~/SsmBundles

dombasedir=${Prefixes}
domreldir=${bndlname%%/*}/d/${namex}${name0}/${name0}_${namev}-s
domdestdir=${dombasedir}/$domreldir

if [[ ! -f ${Prefixes}/${bndlname}.bndl ]] ; then
   echo "ERROR: no such bundl: ${bndlname}"
   echo ${Prefixes}/${bndlname}.bndl
   exit 1
fi

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

depotdir=~/SsmDepot
if [[ ! -d $depotdir ]] ; then
   depotdir=$TMPDIR/SsmDepot
   $myecho mkdir -p $TMPDIR/SsmDepot 2>/dev/null
fi


if [[ ! -d ${domdestdir%/*} ]] ; then
   toto=${domdestdir%/*}
   $myecho chmod u+w ${toto%/*}
   $myecho mkdir -p ${domdestdir%/*} 2>/dev/null
   $myecho chmod u-w ${toto%/*}
fi
$myecho chmod u+w ${domdestdir%/*}
ssm created -d ${domdestdir} --sources $depotdir --force
$myecho chmod u-w ${domdestdir%/*}
$myecho cd ${domdestdir}

myrepublish() {
   _item=$1
   _mylist="$(s.resolve_ssm_shortcuts ${_item} 2>/dev/null | grep -v '++@')"
   for _item2 in $_mylist ; do
      #echo "### ${_item2}"
      if [[ x"${_item2##*.}" != x"sh" \
         && x$_item2 != xPrepend \
         && x$(echo $_item2 | grep devtools) == x \
         && x$(echo $_item2 | grep mpich2) == x ]] ; then
         _mydom="${_item2##*@}"
         _mypkglist="${_item2%%@*}"
         _forcepublish=0
         [[ x$_mydom == x$_mypkglist ]] && _mypkglist=""
         if [[ x$_mypkglist == x ]] ; then
            _mypkglist="$(ssm listd -d $_mydom -o name)"
         else
            _forcepublish=1
         fi
         if [[ x$_mypkglist == x ]] ; then
            echo '#IGNORE cannot find pkg list:' $_mydom
         fi

   for mypkg in $_mypkglist ; do
      mydom="$(ssm listd   -d $_mydom -p $mypkg -o install_domain)"
      mystate="$(ssm listd -d $_mydom -p $mypkg -o state | tr 'A-Z' 'a-z')"
      if [[ x$mystate == xp ||  x$mystate == xip || $_forcepublish == 1 ]] ; then
         echo "#== Publish ${mypkg}@$mydom"
         $myecho ssm publish --yes --force -P $domdestdir -d $mydom -p $mypkg
      else
         echo '#IGNORE' $mystate $mypkg $mydom
      fi
   done

         # for _mypkg0 in $_mypkglist ; do
         #    _mylist2="$(ls -d ${_mydom}/${_mypkg0}*)"
         #    for _mypkgdom in $_mylist2 ; do
         #       _mydom="${_mypkgdom%/*}"
         #       _mypkg="${_mypkgdom##*/}"
         #       if [[ x$_mypkg != xrpncomm_3xx_multi && x$(echo $_mypkg | grep domconfig) == x ]] ; then
         #          echo "#== Publish ${_mypkg}@$_mydom"
         #          $myecho ssm publish --yes --force -P $domdestdir -d $_mydom -p $_mypkg
         #       else
         #          echo "# Skipping ${_mypkg}@$_mydom"
         #       fi
         #    done
         # done
      else
         echo "# Skipping $(echo ${_item2%.*} | sed 's|/home/ordenv/ssm-domains/ssm-setup-1.0/dot-profile-setup_1.0_multi/notshared/data/ssm_domains/||')"
      fi
   done
}

mylist="$(cat ${Prefixes}/${bndlname}.bndl)"

prelist=""
postlist=""
prepost=0
for item in $mylist ; do
   is_sh=0
   [[ -r ${Prefixes}/${item#*:}.sh ]] && is_sh=1
   item4=$(echo ${item#*:} | cut -d@ -f2 | cut -c1-4)
   is_ok=0
#   [[ x$item4 == xENV/ || x$item4 == xGEM/ || x$item4 == xCMDN ]] && is_ok=1
   [[ x$item4 == xENV/ || x$item4 == xGEM/ ]] && is_ok=1
#   item4=$(echo ${item#*:} | grep /gem-mod/)
#   [[ x$item4 != x ]] && is_ok=0
   [[ x${item#*:} != x${item} ]] && is_ok=0
   #echo "@@@ ${is_sh}$is_ok $item"
   if [[ ${is_sh}$is_ok == 01 ]] ; then
      prepost=1
      #echo == $item
      myrepublish $item #${item#*:}
   else
      if [[ $prepost == 0 ]] ; then
         echo "#== Pre $item"
         prelist="$prelist $item"
      else
         echo "#== Post $item"
         postlist="$postlist $item"
      fi
   fi
done
echo "#== Create new SSM Bundle"
#TODO: install/publish domconfig, modified version to work with published only pkg
$myecho chmod -R a-w ${domdestdir}


bndlpath2=$Prefixes/${bndlname2}.bndl
$myecho chmod u+w ${bndlpath2%/*}
if [[ x$myecho == x ]] ; then
   #echo "$prelist $domreldir $postlist GEM/others/rename2s" > $bndlpath2
   echo "$prelist $domreldir $postlist" | tr ' ' '\n' > $bndlpath2
else
   #echo "$prelist $domreldir GEM/others/rename2s $postlist"
   echo "$prelist $domreldir $postlist \> $bndlpath2"
fi
$myecho chmod u-w ${bndlpath2%/*} $bndlpath2
#$myecho chmod go-r $bndlpath2

exit
