#!/bin/ksh
myself=$(true_path $0)
mybasedir=$(true_path ${myself%/*}/..)
TarCmd="echo tar"
tar --help 1>/dev/null 2>/dev/null && TarCmd=tar
gtar --help 1>/dev/null 2>/dev/null && TarCmd=gtar
gnutar --help 1>/dev/null 2>/dev/null && TarCmd=gnutar

supported_installfile_versions=5
BASE_ARCH=${BASE_ARCH:-`uname -s`}
domconfig_ssm_pkg=domconfig_1.0.0_all

ssm_arch_list="\
 Linux_x86-64:linux26-x86-64\
 Linux:linux26-i686\
 AIX:aix53-ppc-64\
"

DESC='Install/publish a set of ssm domains/pacakges as an ssmbundle; Manages dependencies and install order'
USAGE="USAGE: ${myself##*/} [-h] [-v] \\ \n
                    \t [--bndl /path/to/ssm/bundles] \\ \n 
                    \t [--dom /path/to/ssm/domains] \\ \n
                    \t [--rep /path/to/ssm/repository] \\ \n
                    \t [-f INSTALL_INFO_FILE] \\ \n
                    \t [-n PRE_CANNED_INSTALL_FILE_NAME] \\ \n
                    \t [--overwrite [--rm] ]"


#---- Functions ------------------------------------------------------
#====
usage_long() {
	 toto=$(echo -e $USAGE)
	 more <<EOF
$toto

Options:
    -h, --help : print this help
    -v         : verbose mode
    --bndl /path/to/ssm/bundles
                 Provide a path where to create bundles
                 Note that bundles will be created under 
                 /path/to/ssm/bundles/prefix
                 The prefix is the rel/path/ part in ssm_bndl_name 
                 given in the install file below
    --dom  /path/to/ssm/domain
                 Provide a path where to create domains and install pacakges
                 Note that domains will be created under 
                 /path/to/ssm/domain/release
    --rep /path/to/ssm/repository
                 Provide a path where to look for ssm pacakge to install
    -f INSTALL_INFO_FILE
                 Install file name, full path (see below for its format)
                 Option -f OR -n should be provided, noth both
    -n PRE_CANNED_INSTALL_FILE_NAME
                 Install file name for installer provided pre-set files
                 These files are listed in the mod installed pck sub dir
                 ./share/ssm-bundles/
    --overwrite : will install above existing domains/packages and bundles
                  Otherwise dom/pacakge will not be re-installed
                  and bundles will not be re-created
    --rm        : will remove domains/packages and bundles before [re-]installing
                  --rm implies (super-seed) --overwrite

INSTALL_INFO_FILE should look like this:
    modinstaller_file_version=5
    ssm_bndl_basename=GEM
    ssm_bndl_name=x/4.2.2
    ssm_model_name=GEMDM
    ssm_bndl_elements="\
      all:pkg1:version \
      multi:pkg2:version2 \
      every:pkg4:GENERIC@version4 \
      every:pkg5@dom1:version5 \
      all:CONFIG@dom1:version5 \
      MSG_DEV
    "
    export ssm_compiler_list="\
     Linux_x86-64:pgi9xx\
     Linux:pgi9xx\
     AIX:xlf12\
    "
    ssm_bndl_externals="${ssm_compiler_list} rmnlib-dev"
    ssm_bndl_externals_post=""

with these recognized tokens:
  all: ssm all arch
  multi: ssm multi arch
  every: loop over recognized ssm arch... use with GENERIC
  GENERIC: will install ssm pkg pgk_V999V_ARCH as @version
  CONFIG: instal the domain config scripts in the domain
  pkg@dom: install pkg name in dom [pkg@pkg == pkg]
  MSG_DEV: add the devloppement msg file at end of bundle
  MSG_REL: add the release msg file at end of bundle

Defaults are:
    bundledir=~/SsmBundles
    domainsdir=~/SsmDomains
    depotdir=~/SsmDepot
EOF
}

#====
exit_on_error() {
	 echo -e $1
	 echo
	 echo -e $USAGE
	 echo "---- ABORT ----"
	 exit 1

}


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
	 _pkg0=$2
	 _pkg=$3
	 _wkdir=$4
	 _here=`pwd`
	 mkdir -p $_wkdir
	 cd $_wkdir
	 if [[ x$_pkg0 == x$_pkg ]] ; then
		  $TarCmd xzf $depotdir/${_pkg}.ssm ${_pkg}/./.ssm.d/ 2>/dev/null
	 else
		  $TarCmd xzf $depotdir/${_pkg0}.ssm 2>/dev/null
		  mv ${_pkg0} ${_pkg}
		  $TarCmd czf ${_pkg}.ssm ./${_pkg}
	 fi

	 for _dep in $(cat ${_pkg}/.ssm.d/dep-domains.txt 2>/dev/null) ; do
		  . s.ssmuse.dot $_dep
		  #TODO: check if ok
	 done
	 for _dep in $(cat ${_pkg}/.ssm.d/dep-domains-${BASE_ARCH}.txt 2>/dev/null) ; do
		  . s.ssmuse.dot $_dep
		  #TODO: check if ok
	 done
	 for _dep in $(cat ${_pkg}/.ssm.d/dep-pkg.txt 2>/dev/null) ; do
		  . s.ssmuse.dot ${_dep}@$_dom
		  #TODO: check if ok
	 done
	 for _dep in $(cat ${_pkg}/.ssm.d/dep-pkg-${BASE_ARCH}.txt 2>/dev/null) ; do
		  . s.ssmuse.dot ${_dep}@$_dom
		  #TODO: check if ok
	 done

	 if [[ -x ${_pkg}/.ssm.d/pre-install ]] ; then
		  set -x
		  #${_pkg}/.ssm.d/pre-install $_dom $_dom/$_pkg
		  ${_pkg}/.ssm.d/pre-install $(pwd) $(pwd)/$_pkg
		  if [[ $? -ne 0 ]] ; then
				echo "ERROR: problem in pre-install check for pkg ${_pkg}"
				glbstatus=1
		  fi
		  set +x
	 fi

	 rm -rf ${_wkdir}/${_pkg:-__scrap__}/*
	 cd $_here
	 #TODO: should we add the dep to externals in bundle?
}

#====
installpkg() {
	 _dom=$1
	 _pkg0=$2
	 _pkg=$3
	 _dom2=$(true_path $_dom)
	 _exitstatus=0
	 if [[ x$(need_install $_dom2 $_pkg) == x1 ]] ; then
		  _tmpdir=${TMPDIR:-/tmp}/$$
		  mkdir -p $_tmpdir
		  preinstallpkg $_dom2 $_pkg0 $_pkg $_tmpdir
		  if [[ x$glbstatus == x0 ]] ; then
				[[ x$overwrite == x1 ]] && chmod -R 755 $_dom2/$_pkg
				[[ x$remove == x1 ]] && rm -rf $_dom2/$_pkg
				if [[ x$_pkg0 == x$_pkg ]] ; then
					 ssm install $verbose -y -d $_dom2 -p $_pkg
				    #_exitstatus=$?
				else
					 ssm install $verbose -y -d $_dom2 -p $_pkg --repositoryUrl $_tmpdir
				    #_exitstatus=$?
				fi
		  fi
	 else
		  echo "SSM Pkg already installed -- NOT ReInstalling"
	 fi
	 [[ x$(need_install $_dom2 $_pkg) == x1 ]] && _exitstatus=1
	 ssm publish $verbose -y -d $_dom2 -p $_pkg
    #TODO: check if ok
	 chmod -R 555 $_dom2/$_pkg
	 if [[ x$_exitstatus != x0 ]] ; then
		  echo "ERROR: problem installing $_pkg@$_dom"
		  glbstatus=$_exitstatus
	 fi
}

#====
install_msgfile() {
	 _rel_or_dev=$1
	 msgfile=$mybasedir/share/msg-${_rel_or_dev}-version.sh
	 _mydom=${bndl_dir1}/${bndl_name}_msg
	 mymsgfile=${_mydom}.sh
	 mkdir_tree $bundledir ${bndl_dir0}/${ssm_dom_link_name}/${bndl_dir1}
	 chmod 755 $bundledir/${bndl_dir0}/${ssm_dom_link_name}/${bndl_dir1}
	 [[ x$overwrite == x1 ]] && chmod 755 $bundledir/${bndl_dir0}/${ssm_dom_link_name}/$mymsgfile
	 cat $msgfile | \
		  sed "s:__BASEDIR__:${bundledir}:g" | \
		  sed "s:__BUNDLENAME__:${ssm_bndl}:g" | \
		  sed "s:GEMDM:${ssm_model_name:-GEMDM}:g" \
		  > $bundledir/${bndl_dir0}/${ssm_dom_link_name}/$mymsgfile
	 chmod 555 $bundledir/${bndl_dir0}/${ssm_dom_link_name}/$mymsgfile
	 chmod 555 $bundledir/${bndl_dir0}/${ssm_dom_link_name}/${bndl_dir1}
	 echo $_mydom
}

#====
load_dep() {
	 myexternal=$1
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
		  . s.ssmuse.dot ${mypkg}$mydom
	 fi
}

#==== 
mkdir_tree() {
	 _mybasedir=$1
	 _myrelpath=$2
	 _mydir=${_mybasedir}
	 _mysubdirlist=$(echo ${_myrelpath} | tr '/' ' ')
	 for _mysubdir in ${_mysubdirlist} ; do
		  _mydir=${_mydir}/${_mysubdir}
		  if [[ ! -r ${_mydir} ]] ; then
				chmod 755 ${_mydir%/*}
				mkdir -p ${_mydir}
				chmod 555 ${_mydir%/*} ${mydir}
		  fi
	 done
}


#==== Return parsed element in canonical format arch:pkg_ver_arch|pkg_ver_arch@dom
parse_ssm_bndl_element(){
	 _item=$1
	 myarch=${_item%%:*}
	 myarchname=${_item%:*}
	 myname=${myarchname#*:}
	 mydom=${myname#*@}
	 mypkg=${myname%@*}
	 myversion=${_item##*:}
	 myversion_g=${myversion%@*}
	 myversion_xn=${myversion#*@}
	 myversion_x=${myversion_xn%/*}
	 myversion_n=${myversion_xn##*/}

	 myarch_ssm_g=""
    case x$myarch in
        (xevery) myarch_ssm_g=ARCH;;
        (xMSG_DEV) myarch="msg:dev|";;
        (xMSG_REL) myarch="msg:rel|";;
    esac

	 myarch2=$myarch
	 [[ "x${myarch}" == "xevery" ]] && myarch2=$BASE_ARCH
	 myarch_ssm=${myarch}
	 for _arch in $ssm_arch_list ; do
		  _arch1=$(echo ${_arch} | cut -d":" -f1)
		  _arch2=$(echo ${_arch} | cut -d":" -f2)
		  [[ "x${myarch2}" == "x${_arch1}" ]] && myarch_ssm=${_arch2}
	 done
	 [[ x$myarch_ssm_g == x ]] && myarch_ssm_g=$myarch_ssm
	 mycomponent_pkg="${mypkg}_${myversion_n}_${myarch_ssm}"
	 if [[ x$mypkg == xCONFIG ]] ; then
		  mycomponent_pkg=${domconfig_ssm_pkg}
		  myarch=all
	 fi
	 mycomponent_pkg_g=${mycomponent_pkg}
	 [[ x$myversion_g == xGENERIC ]] && mycomponent_pkg_g="${mypkg}_V999V_${myarch_ssm_g}"
	 mycomponent_dom=${mydom}_${myversion_n}
	 mycomponent="${myarch}:${mycomponent_pkg_g}|${mycomponent_pkg}@${mycomponent_dom}"

	 if [[ x"$(echo $_item | cut -c1)" == x"#" ]] ; then
		  echo
	 else
		  echo $mycomponent
	 fi
}

#==== chmod on all dom non pkg subdir
chmod_dom() {
	 _perm=$1
	 _dom=$2
	 chmod $_perm $_dom
	 for subdir in $(ls -d etc lib bin share all* aix* linux* 2>/dev/null) ; do
		  chmod -R $_perm ${_dom%/}/${subdir%/}
	 done
}

#---- Inline Options -------------------------------------------------
verbose=""
bundledir=~/SsmBundles
domainsdir=~/SsmDomains
depotdir=~/SsmDepot
installfile=""
installname=""
overwrite=0
remove=0
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
		  (-r) depotdir=$2 ; shift ;;
		  (--overwrite) overwrite=1 ;;
		  (--rm) remove=1 ;;
        -*) echo "ERROR: Unrecognized option $1"
            echo -e $USAGE
            echo "---- ABORT ----"
            exit 1;;
        *) posargs="$posargs $1" ; ((posargsnum=posargsnum+1));;
    esac
    shift
done
bundledir=${bundledir%/}
domainsdir=${domainsdir%/}

[[ x$remove = x1 ]] && overwrite=1

if [[ x$installfile == x && x$installname == x ]] ; then
	 exit_on_error "ERROR: Need to provide at least -f or -n"
fi

if [[ x$installname != x ]] ; then
	 cannedbundledir=${mybasedir}/share/ssm-bundles
	 installfile=""
	 [[ -r $cannedbundledir/${installname}.txt ]] && installfile="$cannedbundledir/${installname}.txt"
	 [[ -r ~/.ssm-bundles/${installname}.txt ]] && installfile="~/.ssm-bundles/${installname}.txt"
	 if [[ ! -r $installfile ]] ; then
		  exit_on_error "ERROR: Cannot find installation file named: $installname"
	 fi
fi

if [[ ! -r $installfile ]] ; then
	 exit_on_error "ERROR: Cannot read installation file: $installfile"
fi


#---- Read Config File -----------------------------------------------
echo "==== Load Install File"
modinstaller_file_version=-1
ssm_bndl_basename=ENV
ssm_bndl_name=NONE
ssm_model_name=GEMDM
ssm_bndl_elements=""
ssm_bndl_externals=""
ssm_bndl_externals_post=""
ssm_compiler_list=""
ssm_dom_link_name=d

set -x
. $installfile
set +x
ssm_bndl_basename=${ssm_bndl_basename%/}
ssm_bndl_basename=${ssm_bndl_basename:-.}
export ssm_bndl=${ssm_bndl_basename}/$ssm_bndl_name
export ssm_compiler_list

okfile=0
for item in $supported_installfile_versions ; do
	 [[ x$modinstaller_file_version == x$item ]] && okfile=1
done

if [[ okfile -eq 0 ]] ; then
	 exit_on_error "ERROR: Installation File NOT Compatible: $installfile\nVersion='$modinstaller_file_version' [Supported:$supported_installfile_versions]"
fi

if [[ x$ssm_bndl_name == xNONE || x$ssm_bndl_name == x ]] ; then
	 exit_on_error "ERROR: Should at least provide ssm_bndl_name in $installfile"
fi

if [[ x$ssm_bndl_elements == x ]] ; then
	 exit_on_error "ERROR: Should at least provide ssm_bndl_elements in $installfile"
fi

#- Define a few var for backward compatibility
for item in $ssm_compiler_list ; do
	 if [[ "x${item%:*}" == "xAIX" ]] ; then
		  export ssm_AIX_COMPILER="${item#*:}"
	 elif [[ "x${item%:*}" == "xLinux" ]] ; then
		  mycomp="$(echo ${item#*:} | sed 's/-old//')"
		  export ssm_Linux_COMPILER="$mycomp"
	 elif [[ "x${item%:*}" == "xLinux_x86-64" ]]; then
		  mycomp="$(echo ${item#*:} | sed 's/-old//')"
		  export ssm_Linux_x86_64_COMPILER="$mycomp"
	 fi
done

#---- Create basic dir tree ------------------------------------------
echo "==== Create basic dir tree"
#Example for ssm_bndl=GEM/x/4.2.0
#domainsdir/release/GEM/x/4.2.0
#bundledir/GEM
#bundledir/GEM/x/4.2.0
#bundledir/GEM/d      -> domainsdir/release
#bundledir/GEM/d/x/4.2.0

bndl_dir0=${ssm_bndl_basename}
bndl_dir1=${ssm_bndl_name%/*}
bndl_name=${ssm_bndl##*/}
[[ x$bndl_dir0 == x$bndl_name ]] && bndl_dir0='.'
[[ x$bndl_dir0 == x$bndl_dir1 ]] && bndl_dir1='.'
#bndl_dir1d=${ssm_dom_link_name}/${bndl_dir1}

bundledir=${bundledir%/}
domainsdir=${domainsdir%/}
mkdir_tree $domainsdir ${bndl_dir0}
mkdir_tree $bundledir ${bndl_dir0}/${bndl_dir1}
#mkdir_tree $bundledir ${bndl_dir0}/${bndl_dir1d}
#mkdir_tree $bundledir ${bndl_dir0}/${ssm_dom_link_name}
if [[ ! -r $bundledir/${bndl_dir0}/${ssm_dom_link_name} ]] ; then
	 chmod 755 $bundledir/${bndl_dir0}
	 ln -s $domainsdir/${bndl_dir0} $bundledir/${bndl_dir0}/${ssm_dom_link_name}
	 chmod 555 $bundledir/${bndl_dir0}
fi

isok=1
for item in $domainsdir/${bndl_dir0} $bundledir/${bndl_dir0}/${bndl_dir1} $bundledir/${bndl_dir0}/${ssm_dom_link_name} ; do
	 chmod 755 $item
	 [[ ! -w $item ]] && isok=0
	 chmod 555 $item
done
[[ ! -r $depotdir ]] && isok=0

if [[ x$isok == x0 ]] ; then
	 exit_on_error "\
ERROR: all of bundledir, domaindir, depotdir should be writable/readable: \n\
bundledir  = ${bundledir}/${bundlesubdir} \n\
domainsdir = $domainsdir \n\
depotdir   = $depotdir \n\
"
fi


#---------------------------------------------------------------------
cat <<EOF

Installing ${bndl_dir0}/${bndl_dir1}$bndl_name with
    bundlename = $bndl_name
    bundledir  = ${bundledir}
    subdir0    = ${bndl_dir0}
    subdir1    = ${bndl_dir1}
    domainsdir = $domainsdir
    depotdir   = $depotdir

List of items to install:
EOF
for item in $ssm_bndl_elements; do
	 echo "* $item"
done
echo

#---- Load externals -------------------------------------------------
echo "==== Loading dependencies"
for myexternal in $ssm_bndl_externals ; do
	 load_dep $myexternal
    #TODO: check if ok
done

#---- Install pkgs ----------------------------------------------------
bndl_list=""
msg_bndl=""
glbstatus=0
for item in $ssm_bndl_elements ; do
	 item2=$(parse_ssm_bndl_element $item)

	 echo
	 echo "==== $item"
	 echo "==== $item2"
	 echo

	 if [[ x$item2 != x ]] ; then
		  myarch=${item2%%:*}:
		  myinstall=${item2##*:}
		  mydom=${myinstall##*@}
		  mypkg12=${myinstall%%@*}
		  mypkg0=${mypkg12%%\|*}
		  mypkg=${mypkg12##*\|}

		  if [[ "x$myarch" == "xmsg:" ]] ; then

				myarch="all:"
				mydom=$(install_msgfile ${mypkg0})

		  else 

				if [[ "x$myarch" == "xall:" || "x$myarch" == "xmulti:" || "x$myarch" == "xevery:" || "x$myarch" == "x${BASE_ARCH}:" ]] ; then

					 dom_dir_rel=${bndl_dir0}/${mydom}
					 dom_dir_abs=$domainsdir/${dom_dir_rel}

				    #---- create domain
					 mkdir_tree $domainsdir ${dom_dir_rel}
					 chmod 755 ${dom_dir_abs}
					 [[ ! -e ${dom_dir_abs}/etc/ssm.d ]] && s.ssm-creat -d ${dom_dir_abs} -r $depotdir
					 chmod 555 ${dom_dir_abs}
					 if [[ ! -d ${dom_dir_abs}/etc/ssm.d ]] ; then
						  exit_on_error "ERROR: probleme with domain ${dom_dir_rel} [in ${domainsdir}]"
					 fi
					 
				    #---- install pkg
					 if [[ ! -r $depotdir/${mypkg0}.ssm ]] ; then
						  exit_on_error "ERROR: pkg not found, cannot install: $depotdir/${mypkg0}.ssm as ${mypkg}"
					 fi
					 set -x
					 chmod_dom 755 ${dom_dir_abs}
					 installpkg ${dom_dir_abs} $mypkg0 $mypkg
					 chmod_dom 555 ${dom_dir_abs}
					 set +x
					 if [[ x$glbstatus != x0 ]] ; then
						  exit_on_error "ERROR: problem installing $mypkg@$dom_dir_rel"
					 fi
					 
					 . s.ssmuse.dot ${mypkg}@${dom_dir_abs}

				    #---- if dom not visible from bundle_dir, link it
					 if [[ ! -r $bundledir/${bndl_dir0}/${ssm_dom_link_name}/$mydom ]] ; then
						  mydomdir=${mydom%/*}
						  [[ x$mydomdir == ${mydom} ]] && mydomdir=.
						  domlink_dir_rel=${bndl_dir0}/${ssm_dom_link_name}/$mydomdir
						  mkdir_tree $bundledir $domlink_dir_rel
						  chmod 755 $bundledir/$domlink_dir_rel
						  ln -s ${dom_dir_abs} $bundledir/$domlink_dir_rel/${mydom##*/}
						  chmod 555 $bundledir/$domlink_dir_rel
					 fi
					 
				fi

		  fi

		  #---- Add dom to bndl_list [if not already in]
		  echo "adding ($mydom) to bundle as (${bndl_dir0}/${ssm_dom_link_name}/${mydom})"
		  myarch2=${BASE_ARCH}
		  inlist=0
		  mydom2=${bndl_dir0}/${ssm_dom_link_name}/${mydom}
		  for item in $bndl_list ; do
				[[ x$item == x${mydom2} || \
					 x$item == x${myarch2}:${mydom2} ]] && \
					 inlist=1
		  done
		  if [[ inlist -eq 0 ]] ; then
				[[ "x$myarch" == "xall:" || "x$myarch" == "xmulti:" || "x$myarch" == "xevery:" ]] && myarch=''
				bndl_list="$bndl_list ${myarch}${mydom2}"
		  fi

	 fi
done


#---- Create Bundle file ---------------------------------------------
echo "==== Creating ${ssm_bndl}.bndl"
if [[ x$overwrite == x1 || ! -r $bundledir/${ssm_bndl}.bndl ]] ; then
	 chmod 755 $bundledir/${bndl_dir0}/${bndl_dir1} 
	 [[ x$overwrite == x1 ]] && chmod 755 $bundledir/${ssm_bndl}.bndl
	 /bin/rm -f $bundledir/${ssm_bndl}.bndl
	 echo "$ssm_bndl_externals $bndl_list $ssm_bndl_externals_post ${msg_bndl}" > $bundledir/${ssm_bndl}.bndl
	 chmod 555 $bundledir/${bndl_dir0}/${bndl_dir1} 
	 chmod 500 $bundledir/${ssm_bndl}.bndl
else
	 cat <<EOF
WARNING: ${ssm_bndl}.bndl already exists.
         It will not be replaced.
EOF

fi

cat <<EOF
WARNING: Extra manual Steps Needed
         Check if installation went ok then and only then do
         chmod 555 $bundledir/${ssm_bndl}.bndl
         cd $bundledir/${bndl_dir0}/${bndl_dir1}
         chmod 755 .
         ln -sf ${bndl_name}.bndl ${bndl_name%.*}.bndl
         chmod 555 .

EOF

exit 0
