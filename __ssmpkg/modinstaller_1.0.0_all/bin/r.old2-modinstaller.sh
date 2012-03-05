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
    modinstaller_file_version=2
    ssm_bndl_name=name #could also be dir/name

    ssm_bndl_externals='extdom1 extdom2 extbundle1 Linux:extdom4 AIX:extdom5'

    ssm_bndl_install_list="\
    #A_Comment_without_spaces\
    pkg1.1@dom1\
    pkg1.2@dom1\
    pkg2.1@dom2\
    #An_other_Comment_without_spaces\
    Linux:pkg2.2@dom2\
    AIX:pkg2.3@dom2\
    Linux:pkg1.3@dom1\
    AIX:pkg1.4@dom1\
    pkg1.5@dom1\
    pkg2.4@dom2\
    pkg2.5@dom2\
    pkg3.1@dom3\
    "

Defaults are:
    bundledir=~/SsmBundles
    domainsdir=~/SsmDomains
    depotdir=~/SsmDepot
EOF
}

#TODO: allow Linux:pkg2.2/pre@dom2 [post?] to precompile api for example

supported_installfile_versions="2"
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

set -x
. $installfile
set +x

okfile=0
for item in $supported_installfile_versions ; do
	 [[ x$modinstaller_file_version == x$item ]] && okfile=1
done

if [[ okfile -eq 0 ]] ; then
	 echo "ERROR: Installation File NOT Compatible: $installfile"
	 echo "Version='$modinstaller_file_version' [Supported:$supported_installfile_versions]"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi


if [[ x$ssm_bndl_name == x ]] ; then
	 echo "ERROR: Should at least provide ssm_bndl_name in $installfile"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi

if [[ x$ssm_bndl_install_list == x ]] ; then
	 echo "ERROR: Should at least provide ssm_bndl_install_list in $installfile"
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1
fi

bundleprefix=${ssm_bndl_name%/*}
[[ x$bundleprefix == x$ssm_bndl_name ]] && bundleprefix=""

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

Installing $ssm_bndl_name with
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
	 tar xf $depotdir/${_pkg}.ssm ${_pkg}/./.ssm.d/ 2>/dev/null

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
	 _dom2=$(true_path $_dom)
	 [[ x$myecho != x ]] && _dom2=$_dom
	 if [[ x$(need_install $_dom2 $_pkg) == x1 ]] ; then
		  preinstallpkg $_dom2 $_pkg
		  [[ x$overwrite == x1 ]] && $myecho chmod -R 755 $_dom/$_pkg
		  [[ x$remove == x1 ]] && $myecho rm -rf $_dom/$_pkg
		  $myecho ssm install $verbose -y -d $_dom2 -p $_pkg
        #TODO: check if ok
	 fi
	 $myecho ssm publish $verbose -y -d $_dom2 -p $_pkg
    #TODO: check if ok
	 $myecho chmod -R 555 $_dom/$_pkg
}

#---------------------------------------------------------------------

#==== use externals dom/bndl [ ARCH:pkg@dom || ARCH:dom || pkg@dom || dom ]
echo "==== Loading dependencies"
for myexternal in $ssm_bndl_externals ; do
	 myarchdom=${myexternal##*@}
	 myarchpkg=${myexternal%%@*}
	 if [[ x$myarchpkg == x$myarchdom ]] ; then #ARCH:dom || dom
		  mypkg=''
		  mydom=${myarchdom##*:}
		  myarch=${myarchdom%%:*}
		  [[ x$myarch == x$mydom ]] && myarch='all'
	 else #ARCH:pkg@dom || pkg@dom
		  mypkg=''
		  mydom=${myarchdom}
		  mypkg=${myarchpkg##*:}
		  myarch=${myarchpkg%%:*}
		  [[ x$myarch == x$mypkg ]] && myarch='all'
	 fi
	 if [[ x$myarch == xall || x$myarch == x${BASE_ARCH} ]] ; then
		  [[ x$mypkg != x ]] && mypkg="${mypkg}@"
		  $myecho . s.ssmuse.dot ${mypkg}$mydom
	 fi
    #TODO: check if ok
done


#==== Create domaines and install packages
bndl_list=""
for myinstall in $ssm_bndl_install_list ; do
	 if [[ x"$(echo $myinstall | cut -c1)" != x"#" ]] ; then

		  mydom=${myinstall##*@}
		  myarchpkg=${myinstall%%@*}
		  mypkg=${myarchpkg##*:}
		  myarch=${myarchpkg%%:*}
		  [[ x$myarch == x$mypkg ]] && myarch='all'
		  if [[ x$myarch == xall || x$myarch == x${BASE_ARCH} ]] ; then

				echo "==== $myarch:$mypkg@$mydom"
				$myecho cd $domainsdir
				[[ ! -r $mydom ]] && $myecho s.ssm-creat -d $mydom -r $depotdir
				if [[ ! -d $mydom/etc/ssm.d && x$myecho == x ]] ; then
					 echo "ERROR: probleme with domain $mydom [in ${domainsdir}]"
					 echo "---- ABORT ----"
					 exit 1
				fi
				$myecho cd $domainsdir$mydom
				if [[ ! -r $depotdir/${mypkg}.ssm ]] ; then
					 echo "ERROR: pkg not found, cannot install: $depotdir/${mypkg}.ssm"
					 echo "---- ABORT ----"
					 exit 1
				fi
				installpkg ${domainsdir}$mydom $mypkg
		      #TODO: check if ok
				$myecho . s.ssmuse.dot ${mypkg}@${domainsdir}$mydom
		      #TODO: check if ok

				$myecho cd $bundledir/$bundleprefix
				[[ x$remove == x1 ]] && $myecho rm -f  $mydom
				$myecho ln -s ${domainsdir}$mydom $mydom 2>/dev/null

		  fi

		  myarch2=""
		  [[ x$myarch != xall ]] && myarch2="${myarch}:"
		  inlist=0
		  for item in $bndl_list ; do
				[[ x$item == x$bundleprefix/$mydom || \
					 x$item == x${myarch2}$bundleprefix/$mydom ]] && inlist=1
		  done
		  [[ inlist -eq 0 ]] && \
				bndl_list="$bndl_list ${myarch2}$bundleprefix/$mydom"

	 fi
done


#==== Create bundles
echo "==== Create bundles"
$myecho cd $bundledir

[[ x$overwrite == x1 ]] && $myecho rm -f ${ssm_bndl_name}.bndl
if [[ x$myecho == x ]] ; then
	 echo "$ssm_bndl_externals $bndl_list" > ${ssm_bndl_name}.bndl
else
	 $myecho echo "$ssm_bndl_externals $bndl_list" \> ${ssm_bndl_name}.bndl
fi

$myecho chmod 555 $bundleprefix/*


exit 0
