#!/bin/ksh
# @Object: Update files, dirs and links in Build tree for locally modified source
# @Author: S.Chamberland
# @Date:   March 2014
. .rdebase.inc.dot

## Help
DESC='Update files, dirs and links in Build dir for locally modified source'
USAGE="USAGE: ${MYSELF} [-h] [-v] [-f]"
usage_long() {
   toto=$(echo -e $USAGE)
   more <<EOF
$DESC

$toto

Options:
    -h, --help     : print this help
    -v, --verbose  : verbose mode
    -f, --force    : force update build links

EOF
}

rde_exit_if_not_rdetopdir

## Inline Args
myforce=0
while [[ $# -gt 0 ]] ; do
   case $1 in
      (-h|--help) usage_long; exit 0;;
      (-v|--verbose) ((verbose=verbose+1));;
      (-f|--force) myforce=1;;
      (--) shift ;;
      *) if [[ x$1 != x ]] ; then myerror "Option Not recognized: $1";fi;;
   esac
   shift
done

MAKEFILEDEP="${CONST_MAKEFILE_DEP}"
BUILDSRC=$(pwd)/${CONST_BUILDSRC}
BUILDMOD=$(pwd)/${CONST_BUILDMOD}

## ====================================================================

## myrm_obj filename.ftn90
myrm_obj() {
   _filename=$1
   _name=${_filename%.*}
   /bin/rm -f ${_name}.o
   myecho 2 "++ rm ${_name}.o"
}

## myrm_pre filename.ftn90
myrm_pre() {
   _filename=$1
   _name=${_filename%.*}
   _ext=${_filename##*.}
   if [[ x${_ext} == xftn ]] ; then
      /bin/rm -f ${_name}.f
      myecho 2 "++ rm ${_name}.f"
   elif [[ x${_ext} == xftn90 ||  x${_ext} == xcdk90 ]] ; then
      /bin/rm -f ${_name}.f90
      myecho 2 "++ rm ${_name}.f90"
   fi
}

## get_modules_in_file filename.ftn90
get_modules_in_file() {
   # _mylist=""
   # for item in $modnamelist ; do
   #     item2=$(grep -i $item ${1} 2>/dev/null | grep -i module | grep -v '^\s*!' | grep -v '^[c*]')
   #     _mylist="${_mylist} ${_item2}"
   # done
   # echo ${_mylist}
   make -s -f ${MAKEFILEDEP} echo_mydepvar MYVAR=FMOD_LIST_${1##*/}
}

## myrm_mod filename.ftn90
myrm_mod() {
   _filename=$1
   # if [[ x"$(echo ${EXT4MODLIST} | grep '\.${_filename##*.}\ ')" == x ]] ; then
   #    return
   # fi
   _modlist="$(get_modules_in_file ${_filename})"
   for _mymod in ${_modlist} ; do
      for _myfile in $(ls -1 ${BUILDMOD}) ; do
         _myname=$(echo ${_myfile##*/} |tr 'A-Z' 'a-z')
         if [[ x${_myname%.*} == x${_mymod} ]] ; then
            #myecho 2 "++ $(ls ${BUILDMOD}/${_myname})"
            #/bin/rm -f ${_myfile}
            #myecho 2 "++ rm BUILDMOD/${_myfile}"
            /bin/rm -f ${BUILDMOD}/${_myname}
            myecho 2 "++ rm BUILDMOD/${_myname}"
            #myecho 2 "++ $(ls ${BUILDMOD}/${_myname})"
         fi
      done
   done
}

## get_invdep_list filename.ftn90
get_invdep_list() {
   make -s -f ${MAKEFILEDEP} echo_mydepvar MYVAR=INVDEP_LIST_${1}
}


## myrm_dep filename.ftn90
myrm_invdep() {
   _filename=$1
   _invdeplist="$(get_invdep_list ${_filename})"
   for _myobjfile in ${_invdeplist} ; do
      #/bin/rm -f ${_myobjfile%.*}.*
      for _myext in $SRCSUFFIXES ; do
         if [[ -f ${_myobjfile%.*}${_myext} && ! -L ${_myobjfile%.*}${_myext} ]] ; then
            myrm_obj ${_myobjfile%.*}${_myext}
            myrm_pre ${_myobjfile%.*}${_myext}
            myrm_mod ${_myobjfile%.*}${_myext}
            /bin/rm -f ${_myobjfile%.*}${_myext}
            myecho 2 "++ rm ${_myobjfile%.*}${_myext} #for ${_filename}"
         fi
      done
   done
   #TODO: update
   # _deplist="$(get_dep_list ${_filename})"
   # for _item in ${_deplist} ; do
   #     myrm_obj ${_item}
   #     myrm_pre ${_item}
   #     myrm_mod ${_item}
   #     /bin/rm -f ${_item}
   #     myecho 2 "++ rm ${_item}"
   # done
}

##
myrm_bidon() {
   _list="`grep c_to_f_sw *.c 2>/dev/null | cut -d':' -f1`"
   for _item in ${_list} ; do
      /bin/rm -f ${_item%.*}.[co]
      myecho 2 "++ rm ${_item%.*}.[co]"
   done
}

##
myrm_empty() {
   toto=""
   #TOTO: update
}

## ====================================================================
#VALIDEXTWILD="*.F *.F90 *.f *.f90 *.ftn *.ftn90 *.cdk *.cdk90 *.fh* *.inc *.h* *.c *.cpp"
VALIDEXTWILD="$(echo ${VALIDEXT} | sed 's/\./*./g')"
VALIDEXTWILD2="$(echo ${VALIDEXT} | sed 's|\.|include/*.|g')"

mylist="$(ls ${SRC_PATH_FILE} Makefile.build.mk ${MAKEFILEDEP} ${VALIDEXTWILD} ${VALIDEXTWILD2} 2>/dev/null | sort)"

if [[ "x${BUILDSRC}" == "x" || ! -d ${BUILDSRC:-__NO_SUCH_DIR__} ]] ; then
   echo "ERROR: BUILDSRC=${BUILDSRC} Not Defined or Not found" 1>&2
   exit 1
fi

cd ${BUILDSRC}
if [[ $? != 0 ]] ; then
    echo "ERROR: Problem changing to BUILDSRC=${BUILDSRC}" 1>&2
   exit 1
fi

if [[ x"$(true_path $(pwd))" == x"$(true_path ${ROOT})" ]] ; then
   echo "ERROR: BUILDSRC=${BUILDSRC} is the same as ROOT=${ROOT}, please check \$storage_model and re-do 'rdemklink -f -v'" 1>&2
   exit 1
fi


## Checking changes status
echo ${mylist} > ${TMPDIR}/.rdesrcusrls
#echo ${mylist2} > ${TMPDIR}/.rdesrcusrls2
# diff ${TMPDIR}/.rdesrcusrls2 .rdesrcusrls > /dev/null 2>&1 \
# && \
# diff ${TMPDIR}/.rdesrcusrls .rdesrcusrls > /dev/null 2>&1
diff ${TMPDIR}/.rdesrcusrls .rdesrcusrls > /dev/null 2>&1
if [[ x$? == x0 && ${myforce} == 0 ]] ; then
   /bin/rm -f Makefile
   ln -s Makefile.build.mk Makefile
   myecho 1 "++ rdeupdate_build_links: Nothing changed since last rdeupdate"
   exit 0
fi
myecho 2 "++ Updating build links"

## Remove dangling links and obsolete files
for item in * ; do
   if [[ -L ${item} ]] ; then
      #if [[ ! -f ${ROOT}/${item} ]] ; then
      if [[ ! -f $(readlink ${item}) ]] ; then
         myecho 2 "++ removed ${item}"
         myrm_obj ${item}
         myrm_pre ${item}
         myrm_mod ${item}
         myrm_invdep ${item} #when ${item} is .cdk or .cdk90... need to remove .o, .mod of files having it as a dependency, use make_cdk for that
         /bin/rm -f ${item}
         rdeco -q ${item}
      fi
   fi
done
myrm_bidon
myrm_empty

## re-make links to source files
for item in ${mylist} ; do
   myecho 2 "++ ln -s $item"
   /bin/rm -f ${item##*/}
   ln -s ${ROOT}/${item} ${item##*/}
done

/bin/rm -f Makefile
ln -s Makefile.build.mk Makefile

mv $TMPDIR/.rdesrcusrls . 2>/dev/null
