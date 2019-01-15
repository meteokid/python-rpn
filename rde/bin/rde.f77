#!/bin/ksh
#
# s.f77

# set EC_LD_LIBRARY_PATH and EC_INCLUDE_PATH using LD_LIBRARY_PATH
#export EC_LD_LIBRARY_PATH=`s.generate_ec_path --lib`
#export EC_INCLUDE_PATH=`s.generate_ec_path --include`

COMPILING_FORTRAN=YES
. rde.get_compiler_rules.dot

if [[ -n $Verbose ]] ; then
   cat <<EOF
${FC:-ERROR_FC_undefined} ${SourceFile} $FC_options ${FFLAGS} \\
	$(s.prefix "${Dprefix}" ${DEFINES} ) \\
	$(s.prefix "${Iprefix}" ${INCLUDES} ${EC_INCLUDE_PATH}) \\
	$(s.prefix "${Lprefix}" ${LIBRARIES_PATH} ${EC_LD_LIBRARY_PATH}) \\
	$(s.prefix "${lprefix}" ${LIBRARIES} ${SYSLIBS} ) \\
	"$@"
EOF
fi

${FC:-ERROR_FC_undefined} ${SourceFile} $FC_options ${FFLAGS} \
	$(s.prefix "${Dprefix}" ${DEFINES} ) \
	$(s.prefix "${Iprefix}" ${INCLUDES} ${EC_INCLUDE_PATH}) \
	$(s.prefix "${Lprefix}" ${LIBRARIES_PATH} ${EC_LD_LIBRARY_PATH}) \
	$(s.prefix "${lprefix}" ${LIBRARIES} ${SYSLIBS} ) \
	"$@"
