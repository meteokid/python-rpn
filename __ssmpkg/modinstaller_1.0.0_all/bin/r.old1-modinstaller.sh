#!/bin/ksh

DESC='Install/publish a set of ssm domains/pacakges as an ssmbundle; Manages dependencies and install order'
USAGE="USAGE: ${0##*/} [-h] [-v] \\ \n
                    \t [--bndl /path/to/ssm/bundles] \\ \n 
                    \t [--dom /path/to/ssm/domains] \\ \n
                    \t [--rer /path/to/ssm/repository] \\ \n
                    \t [-f INSTALL_INFO_FILE] \\ \n
                    \t [-n PRE_CANNED_INSTALL_FILE_NAME] \\ \n
                    \t [--dry] [--test] [--overwrite [--rm] ]"

usage_long() {
	 toto=$(echo -e $USAGE)
	 more <<EOF
$toto

Options:
    -h, --help : print this help
    -v         : verbose mode
    --bndl /path/to/ssm/bundles
                 Provide a path where to create bundles
    --dom  /path/to/ssm/domain
                 Provide a path where to create domains and install pacakges
    --rep /path/to/ssm/repository
                 Provide a path where to look for ssm pacakge to install
    -f INSTALL_INFO_FILE
                 Install file name, full path (see below for its format)
                 Option -f OR -n should be provided, noth both
    -n PRE_CANNED_INSTALL_FILE_NAME
                 Install file name for installer provided pre-set files
                 These files are listed in the mod installed pck sub dir
                 ./share/ssm-bundles/
    --dry      : make a dry run (print what it will do w/o doing it)
    --test     : install bundle and domains/pacakge under test subdirs
                 This will create a test dir under the bndl and dom dirs
                 bundles can then be used by updating the SSM_SHORTCUT_PATH
                     export SSM_SHORTCUT_PATH="/path/to/ssm/bundles/GEM/test:$SSM_SHORTCUT_PATH"
                     . s.ssmuse.dot mybundle
    --overwrite : will install above existing domains/packages and bundles
                  Otherwise dom/pacakge will not be re-installed
                  and bundles will not be re-created
    --rm        : will remove domains/packages and bundles before [re-]installing
                  --rm implies (super-seed) --overwrite

INSTALL_INFO_FILE should look like this:
    bundlename=name #could also be dir/name
    archlist="Linux AIX"

    externals='extdom1 extdom2 extbundle1'
    externals_Linux='extdom4 extbundle12'
    externals_AIX='extdom5 extbundle15'

    domains='dom1 dom2'

    dom1='pkg1 pkg2'
    dom2='pkg1 pkg3'

    domains_Linux='dom3 dom2'
    dom2_Linux='pkg4'
    dom3_Linux='pkg5'

    domains_AIX='dom5'
    dom5_AIX='pkg7'

Defaults are:
    bundledir=~/SsmBundles
    domainsdir=~/SsmDomains
    depotdir=~/SsmDepot
EOF
}

BASE_ARCH=${BASE_ARCH:-`uname -s`}
domconfig=domconfig_1.0.0_all

myecho="echo" #TODO: make it en empty string when done testing
verbose=""
bundledir=~/SsmBundles
domainsdir=~/SsmDomains
depotdir=~/SsmDepot
installfile=""
installname=""
overwrite=0
remove=0
modtest=""
posargs=""
posargsnum=0
while [[ $# -gt 0 ]] ; do
    case $1 in
        (-h|--help) echo $DESC ; usage_long; exit 0;;
		  (-v) verbose="--verbose" ;;
		  (-f) installfile=$2 ; shift ;;
		  (-n) installname=$2 ; shift ;;
		  (--bndl) bundledir=$2 ; shift ;;
		  (--dom) domainsdir=$2 ; shift ;;
		  (--dry) myecho="echo" ;;
		  (-r) depotdir=$2 ; shift ;;
		  (--overwrite) overwrite=1 ;;
		  (--rm) remove=1 ;;
		  (--test) modtest="GEM/test/";;
        -*) echo "ERROR: Unrecognized option $1"
            echo -e $USAGE
            echo "---- ABORT ----"
            exit 1;;
        *) posargs="$posargs $1" ; ((posargsnum=posargsnum+1));;
    esac
    shift
done

[[ x$remove = x1 ]] && overwrite=1

if [[ x$installfile == x && x$installname == x ]] ; then
	 echo "ERROR: Need to provide at least -f or -n"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi

if [[ x$installname != x ]] ; then
	 cannedbundledir=$(true_path ${0%/*})
	 cannedbundledir=${cannedbundledir%/*}/share/ssm-bundles
	 installfile=""
	 [[ -r $cannedbundledir/${installname}.txt ]] && installfile="$cannedbundledir/${installname}.txt"
	 [[ -r ~/.ssm-bundles/${installname}.txt ]] && installfile="~/.ssm-bundles/${installname}.txt"
	 if [[ ! -r $installfile ]] ; then
		  echo "ERROR: Cannot find installation file named: $installname"
		  echo -e $USAGE
		  echo "---- ABORT ----"
		  exit 1
	 fi
fi

if [[ ! -r $installfile ]] ; then
	 echo "ERROR: Cannot read installation file: $installfile"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi

. $installfile

if [[ x$bundlename == x ]] ; then
	 echo "ERROR: Should at least provide bundlename in $installfile"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi

bundleprefix=${bundlename%/*}
[[ x$bundleprefix == x$bundlename ]] && bundleprefix=""

#bundleprefix=${modtest}${bundleprefix}/
domainsdir=${domainsdir}/${modtest}
bundledir=$bundledir/${modtest}

mkdir -p ${bundledir}$bundleprefix 2>/dev/null
mkdir -p $domainsdir 2>/dev/null

if [[ ! -w ${bundledir}$bundleprefix || ! -w $domainsdir || ! -w $depotdir ]] ; then
	 echo "ERROR: all of bundledir, domaindir, depotdir should be writable: "
	 echo "bundledir  = ${bundledir}$bundleprefix"
	 echo "domainsdir = $domainsdir"
	 echo "depotdir   = $depotdir"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi



if [[ x$myecho != x ]] ; then
	 echo "WARNING: performing a dry run -- will NOT install"
fi
cat <<EOF

Installing $bundlename with
    bundledir=$bundledir
    domainsdir=$domainsdir
    depotdir=$depotdir

EOF

#---------------------------------------------------------------------

#====
need_install() {
	 _dom=$1
	 _pkg=$2
	 _need=0
	 [[ ! -r $_dom/$_pkg ]] && _need=1
	 [[ ! -r $_dom/etc/ssm.d/installed/$_pkg ]] && _need=1
	 [[ x"${_pkg##*_}" == x"multi" ]] && _need=1
	 [[ x$overwrite == x1 ]] && _need=1
	 echo $_need
}

#====
preinstallpkg() {
	 _dom=$1
	 _pkg=$2
	 _here=`pwd`
	 mkdir -p ${TMPDIR:-/tmp/$$}
	 cd ${TMPDIR:-/tmp/$$}
	 tar xf $depotdir/${_pkg}.ssm ${_pkg}/./.ssm.d/

	 for _dep in $(cat ${_pkg}/.ssm.d/dep-domains.txt 2>/dev/null) ; do
		  $myecho . s.ssmuse.dot $_dep
		  #TODO: check if ok
	 done
	 for _dep in $(cat ${_pkg}/.ssm.d/dep-domains-${BASE_ARCH}.txt 2>/dev/null) ; do
		  $myecho . s.ssmuse.dot $_dep
		  #TODO: check if ok
	 done
	 for _dep in $(cat ${_pkg}/.ssm.d/dep-pkg.txt 2>/dev/null) ; do
		  $myecho . s.ssmuse.dot ${_dep}@$_dom
		  #TODO: check if ok
	 done
	 for _dep in $(cat ${_pkg}/.ssm.d/dep-pkg-${BASE_ARCH}.txt 2>/dev/null) ; do
		  $myecho . s.ssmuse.dot ${_dep}@$_dom
		  #TODO: check if ok
	 done

	 if [[ -x ${_pkg}/.ssm.d/pre-install ]] ; then
		  $myecho ${_pkg}/.ssm.d/pre-install $_dom $_dom/$_pkg
		  if [[ $? -ne 0 ]] ; then
				echo "ERROR: problem in pre-install check for pkg ${_pkg}"
				echo "---- ABORT ----"
				exit 1
		  fi
	 fi

	 rm -rf ${TMPDIR:-/tmp/$$}/${_pkg:-__scrap__}/*
	 cd $_here
	 #TODO: should we add the dep to externals in bundle?
}

#====
installpkg() {
	 _dom=$1
	 _pkg=$2
	 echo "==== Package: $_dom/$_pkg"
	 if [[ x$(need_install $_dom $_pkg) == x1 ]] ; then
		  preinstallpkg $_dom $_pkg
		  [[ x$overwrite == x1 ]] && $myecho chmod -R 755 $_dom/$_pkg
		  [[ x$remove == x1 ]] && $myecho rm -rf $_dom/$_pkg
		  $myecho ssm install $verbose -y -d $(true_path $_dom) -p $_pkg
        #TODO: check if ok
	 fi
	 $myecho ssm publish $verbose -y -d $(true_path $_dom) -p $_pkg
    #TODO: check if ok
	 $myecho chmod -R 555 $_dom/$_pkg
}

#---------------------------------------------------------------------

#==== use externals dom/bndl
echo "======== Loading dependencies"
for mydom in $externals $(eval echo \$externals_`echo ${BASE_ARCH}`); do
	 $myecho . s.ssmuse.dot $mydom
    #TODO: check if ok
done

#==== Create domaines and install packages
cd $domainsdir
for mydom in $domains $(eval echo \$domains_`echo ${BASE_ARCH}`); do
	 echo "======== Domain: $mydom"
	 [[ ! -r $mydom ]] && $myecho s.ssm-creat -d $mydom -r $depotdir
	 [[ ! -d $mydom ]] && [[ x$myecho == x ]] && exit 1
	 $myecho cd $mydom
	 mydomname=$(echo $mydom | tr '.' '_')
	 for mypkg in $domconfig $(eval echo $`echo $mydomname`) $(eval echo $`echo $mydomname`_`echo ${BASE_ARCH}`); do
		  if [[ ! -r $depotdir/${mypkg}.ssm ]] ; then
				echo "ERROR: pkg not found, cannot install: $depotdir/${mypkg}.ssm"
				echo "---- ABORT ----"
				exit 1
		  fi
		  installpkg ${domainsdir}$mydom $mypkg
		  #TODO: check if ok
		  $myecho . s.ssmuse.dot ${mypkg}@${domainsdir}$mydom
		  #TODO: check if ok
	 done
	 cd ..
done

#==== Create bundles
echo "======== Create bundles"
$myecho cd $bundledir

[[ x$overwrite == x1 ]] && $myecho rm -f ${bundlename}_ext.bndl
if [[ x$myecho == x ]] ; then
	 echo ${externals} > ${bundlename}_ext.bndl
	 for myarch in ${archlist:-$BASE_ARCH} ; do
		  [[ x$overwrite == x1 ]] && $myecho rm -f ${bundlename}_ext_${myarch}.bndl
		  echo $(eval echo \$externals_`echo ${myarch}`) > ${bundlename}_ext_${myarch}.bndl
	 done
else
	 $myecho echo ${externals} \> ${bundlename}_ext.bndl
	 for myarch in ${archlist:-$BASE_ARCH} ; do
		  [[ x$overwrite == x1 ]] && $myecho rm -f ${bundlename}_ext_${myarch}.bndl
		  $myecho echo $(eval echo \$externals_`echo ${myarch}`) \> ${bundlename}_ext_${myarch}.bndl
	 done
fi

for mydom in $domains ; do
	 [[ x$remove == x1 ]] && $myecho rm -f  $bundleprefix/$mydom
	 $myecho ln -s ${domainsdir}$mydom $bundleprefix/$mydom
done

extlist="${bundlename}_ext"
for myarch in ${archlist:-$BASE_ARCH} ; do
	 extlist="$extlist ${myarch}:${bundlename}_ext_${myarch}"
done

domainslist=""
for mydom in $domains; do
	 domainslist="$domainslist $bundleprefix/$mydom"
done

[[ x$overwrite == x1 ]] && $myecho rm -f ${bundlename}.bndl
if [[ x$myecho == x ]] ; then
	 echo "$extlist $domainslist" > ${bundlename}.bndl
else
	 $myecho echo "$extlist $domainslist" \> ${bundlename}.bndl
fi

$myecho chmod 555 $bundleprefix/*

exit 0
