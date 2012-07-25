#!/bin/ksh
myself=$(true_path $0)
mybasedir=$(true_path ${myself%/*}/..)
. r.misc-fn.dot

_notify=1

supported_installfile_versions=6
BASE_ARCH=${BASE_ARCH:-`uname -s`}
domconfig_ssm_pkg=domconfig_1.0.0_all

ssm_arch_list="\
 Linux_x86-64:linux26-x86-64\
 AIX-powerpc7:aix61-ppc-64\
"
#dryrun=1
dryrun=0

DESC='Install/publish a set of ssm domains/pacakges as an ssmbundle; Manages dependencies and install order'
USAGE="USAGE: ${myself##*/} [-h] [-v] [--allonly]] \\ \n
                    \t [--bndl /path/to/ssm/bundles] \\ \n 
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
    --allonly  : only install package of ssm type _all (not ARCH specific)
    --bndl /path/to/ssm/bundles-domains
                 Provide a path where to create bundles and domains
                 Note that bundles will be created under 
                 /path/to/ssm/bundles/prefix
                 The prefix is the rel/path/ part in ssm_bndl_name 
                 given in the install file below
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
    modinstaller_file_version=${supported_installfile_versions}
    ssm_bndl=GEM/x/4.5.0-a4
    ssm_model_name=GEMDM
    ssm_bndl_elements="\
      pkg1_version_all@domain \
      pkg1_version_ARCH@domain \
      CONFIG@domain \
    "
    ssm_compiler_list="\
      Linux_x86-64:pgi9xx\
      AIX-powerpc7:Xlf13\
    "
    ssm_bndl_externals="${ssm_compiler_list} rmnlib-dev"
    ssm_bndl_externals_post=""

with these recognized tokens:
  CONFIG@domain: instal the domain config scripts in the domain
  pkg1_version_all@domain: install pkg1_version_all in the domain
  pkg1_version_ARCH@domain: install pkg1_version_* in the domain for all know ssm arch

Defaults are:
    bundledir=~/SsmBundles
    depotdir=~/SsmDepot
EOF
}

#====
preinstallpkg() {
	 _dom=$1
	 _pkg0=$2
	 _pkg=$3
	 _wkdir=$4
    _depotdir=${5:-${depotdir:-/dev/null}}
	 _here=`pwd`
	 TarCmd=$(define_TarCmd)
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
	 _pkg=$2
	 _overwrite=${3:-${overwrite:-0}}
	 _dom2=$(true_path $_dom)
	 _exitstatus=0
	 if [[ x$(ssm_pkg_need_install $_dom2 $_pkg $_overwrite) == x1 ]] ; then
		  # _tmpdir=${TMPDIR:-/tmp}/$$
		  # mkdir -p $_tmpdir
		  # preinstallpkg $_dom2 $_pkg0 $_pkg $_tmpdir
		  # if [[ x$glbstatus == x0 ]] ; then
		  #  	[[ x$_overwrite == x1 ]] && chmod -R 755 $_dom2/$_pkg
		  #  	[[ x$remove == x1 ]] && rm -rf $_dom2/$_pkg
		  #  	if [[ x$_pkg0 == x$_pkg ]] ; then
		  #  		 ssm install $verbose -y -d $_dom2 -p $_pkg
		  #  	    #_exitstatus=$?
		  #  	else
		  #  		 ssm install $verbose -y -d $_dom2 -p $_pkg --repositoryUrl $_tmpdir
		  #  	    #_exitstatus=$?
		  #  	fi
		  # fi
       #TODO: preinstallpkg $_dom2 $_pkg $_tmpdir
		 [[ x$_overwrite == x1 ]] && chmod -R 755 $_dom2/$_pkg
		 [[ x$remove == x1 ]] && rm -rf $_dom2/$_pkg
		 ssm install $verbose -y -d $_dom2 -p $_pkg
	 else
		  echo "SSM Pkg already installed -- NOT ReInstalling"
	 fi
	 [[ x$(ssm_pkg_need_install $_dom2 $_pkg $_overwrite) == x1 ]] && _exitstatus=1
	 [[ x$(ssm_pkg_need_publish $_dom2 $_pkg $_overwrite) == x1 ]] && ssm publish $verbose -y -d $_dom2 -p $_pkg
	 [[ x$(ssm_pkg_need_publish $_dom2 $_pkg $_overwrite) == x1 ]] && _exitstatus=1
	 chmod -R 555 $_dom2/$_pkg
	 mylist="$(ls -d $_dom2/$_pkg/src $_dom2/$_pkg/RCS_* $_dom2/$_pkg/include 2>/dev/null)"
	 for item in $mylist ; do
		  find ${item}/ -type f -exec chmod 444 {} \; 2>/dev/null
	 done
	 if [[ x$_exitstatus != x0 ]] ; then
		  echo "ERROR: problem installing/publishing $_pkg@$_dom"
		  glbstatus=$_exitstatus
	 fi
}


#---- Inline Options -------------------------------------------------
verbose=""
bundledir=~/SsmBundles
depotdir=~/SsmDepot
installfile=""
installname=""
overwrite=0
remove=0
posargs=""
posargsnum=0
allonly=0
while [[ $# -gt 0 ]] ; do
    case $1 in
        (-h|--help) echo $DESC ; usage_long; exit 0;;
		  (-v) verbose="--verbose" ;;
		  (-f) installfile=$2 ; shift ;;
		  (-n) installname=$2 ; shift ;;
		  (--bndl) bundledir=$2 ; shift ;;
		  (-r) depotdir=$2 ; shift ;;
		  (--overwrite) overwrite=1 ;;
		  (--allonly) allonly=1 ;;
		  (--rm) remove=1 ;;
        -*) exit_on_error "ERROR: Unrecognized option $1";;
        *) posargs="$posargs $1" ; ((posargsnum=posargsnum+1));;
    esac
    shift
done
bundledir=${bundledir%/}
domainsdir=$bundledir

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
echo
echo "==== Load Install File: $installfile"
echo
modinstaller_file_version=-1
ssm_bndl=
ssm_model_name=GEMDM
ssm_bndl_elements=""
ssm_bndl_externals=""
ssm_bndl_externals_post=""
ssm_compiler_list=""

set -x
. $installfile
set +x
export ssm_bndl ssm_model_name ssm_bndl_elements ssm_bndl_externals ssm_bndl_externals_post ssm_compiler_list 

okfile=0
for item in $supported_installfile_versions ; do
	 [[ x$modinstaller_file_version == x$item ]] && okfile=1
done

if [[ okfile -eq 0 ]] ; then
	 exit_on_error "ERROR: Installation File NOT Compatible: $installfile\nVersion='$modinstaller_file_version' [Supported:$supported_installfile_versions]"
fi

if [[ x$ssm_bndl == x ]] ; then
	 exit_on_error "ERROR: Should at least provide ssm_bndl in $installfile"
fi

if [[ x$ssm_bndl_elements == x ]] ; then
	 exit_on_error "ERROR: Should at least provide ssm_bndl_elements in $installfile"
fi

set_lock


#---------------------------------------------------------------------
cat <<EOF

====

Installing ${ssm_bndl} with
    bundlename = ${ssm_bndl##*/}
    bundledir0 = ${bundledir}
    bundledir  = ${ssm_bndl%/*}
    depotdir   = $depotdir

List of items to install:
EOF
for item in $ssm_bndl_elements; do
	 echo "* $item"
done


#---- Create basic dir tree ------------------------------------------
echo
echo "==== Create basic dir tree"
echo
set -x
mkdir_tree $bundledir ${ssm_bndl%/*}
set +x

isok=1
for item in $bundledir/${ssm_bndl%/*} ; do
	 chmod 755 $item
	 [[ ! -w $item ]] && isok=0
	 chmod 555 $item
done
[[ ! -r $depotdir ]] && isok=0

if [[ x$isok == x0 ]] ; then
	 exit_on_error "\
ERROR: all of bundledir, domaindir, depotdir should be writable/readable: \n\
bundledir  = ${bundledir}/${ssm_bndl%/*} \n\
depotdir   = $depotdir \n\
"
fi

#---- Load externals -------------------------------------------------
echo
echo "==== Loading dependencies"
echo
for myexternal in $ssm_bndl_externals ; do
	 load_ssm_bndl_part $myexternal
    #TODO: check if ok
done

#---- Install pkgs ----------------------------------------------------
bndl_list=""
msg_bndl=""
glbstatus=0

#if [[ x$dryrun == x1 ]] ; then
#	 set -x
#fi

for item in $ssm_bndl_elements ; do
   [[ x$(echo $item | cut -c1) == x# ]] && continue
	echo
	echo "==== $item"

	mydom=${item##*@}
	mypkg123=${item%%@*}
   [[ x$mypkg123 == xCONFIG ]] && mypkg123=$domconfig_ssm_pkg

	mypkg_name=${mypkg123%%_*}
	mypkg_version=$(echo $mypkg123| cut -d_ -f2)
	mypkg_arch=${mypkg123##*_}

	if [[ x$item != x && x$mydom != x && x$mypkg_version != x && x$mypkg_name != x$mypkg_arch ]] ; then
      
	   echo
	   echo "---- Create domain: ${mydom}"
		 #---- create domain
		mkdir_tree ${domainsdir} ${mydom}
      dom_dir_abs=$(true_path ${domainsdir}/${mydom})
		chmod 755 ${dom_dir_abs}
		[[ ! -e ${dom_dir_abs}/etc/ssm.d ]] && s.ssm-creat -d ${dom_dir_abs} -r $depotdir
		chmod 555 ${dom_dir_abs}
		if [[ ! -d ${dom_dir_abs}/etc/ssm.d ]] ; then
			exit_on_error "ERROR: probleme with domain ${mydom} [in ${domainsdir}]"
		fi

	    #---- install pkg
      myarchlist=$mypkg_arch
      if [[ x${mypkg_arch} == xARCH ]] ; then
         myarchlist=""
         for myarch in $ssm_arch_list ; do
            myarchlist="$myarchlist $(echo $myarch | cut -d: -f2)"
         done
      fi
       #TODO: if allonly
      for myarch in $myarchlist ; do
         mypkg=${mypkg_name}_${mypkg_version}_${myarch}
	      echo
	      echo "---- Install Pkg: ${mypkg}"
		   if [[ ! -r $depotdir/${mypkg}.ssm ]] ; then
			   exit_on_error "ERROR: pkg not found, cannot install: $depotdir/${mypkg}.ssm"
		   fi
		   set -x
		   chmod_ssm_dom 755 ${dom_dir_abs}
		   installpkg ${dom_dir_abs} ${mypkg}
		   chmod_ssm_dom 555 ${dom_dir_abs}
			set +x
			if [[ x$glbstatus != x0 ]] ; then
				exit_on_error "ERROR: problem installing $mypkg@$mydom"
			fi
			
			. s.ssmuse.dot ${mypkg}@${dom_dir_abs}

      done

		 #---- Add dom to bndl_list [if not already in]
	   echo
		echo "---- Adding $mydom to ${ssm_bndl}.bndl bundle"
		myarch2=${BASE_ARCH}
		inlist=0
		for item in $bndl_list ; do
			[[ x$item == x${mydom} || \
				x$item == x${myarch2}:${mydom} ]] && \
				inlist=1
		done
		[[ x$inlist == x0 ]] && bndl_list="$bndl_list ${mydom}"
      
   else
		exit_on_error "ERROR: Invalid item $item"
	fi
done


#---- Create Bundle file ---------------------------------------------
echo
echo "==== Creating ${ssm_bndl}.bndl"
echo
if [[ x$overwrite == x1 || ! -r $bundledir/${ssm_bndl}.bndl ]] ; then
	 item=$bundledir/${ssm_bndl}.bndl
	 chmod 755 ${item%/*}
	 [[ x$overwrite == x1 ]] && chmod 755 $bundledir/${ssm_bndl}.bndl
	 /bin/rm -f $bundledir/${ssm_bndl}.bndl
	 echo "$ssm_bndl_externals $bndl_list $ssm_bndl_externals_post" > $bundledir/${ssm_bndl}.bndl
	 chmod 555 ${item%/*}
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

EOF

unset_lock
if [[ x$_notify == x1 ]] ; then
	 send_email_notice OK 'Normal Ending; Extra manual Steps Needed'
fi
exit 0
