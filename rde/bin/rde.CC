#!/bin/ksh
#
# s.CC

# set EC_LD_LIBRARY_PATH and EC_INCLUDE_PATH using LD_LIBRARY_PATH
#export EC_LD_LIBRARY_PATH=`s.generate_ec_path --lib`
#export EC_INCLUDE_PATH=`s.generate_ec_path --include`

#$CPP $CPP_options $CPP_LD_options "$@"
export COMPILING_C=yes
. rde.get_compiler_rules.dot

CC=${CPP:-c++}
CC_options="${CPP_options:-${CC_options}}"
CC_options_NOLD="${CPP_options_NOLD:-${CC_options_NOLD}}"

if [ -n $WILL_LINK ]; then
   if [[ -n $Verbose ]] ; then
      cat <<EOF
${CC:-ERROR_CC_undefined} ${SourceFile} $CC_options ${CFLAGS} \\
		$(s.prefix "" ${DEFINES} ) \\
		$(s.prefix "-I" ${INCLUDES} ${EC_INCLUDE_PATH}) \\
		$(s.prefix "-L" ${LIBRARIES_PATH} ${EC_LD_LIBRARY_PATH}) \\
		$(s.prefix "-l" ${LIBRARIES} ) \\
		"$@"
EOF
   fi
	${CC:-ERROR_CC_undefined} ${SourceFile} $CC_options ${CFLAGS} \
		$(s.prefix "" ${DEFINES} ) \
		$(s.prefix "-I" ${INCLUDES} ${EC_INCLUDE_PATH}) \
		$(s.prefix "-L" ${LIBRARIES_PATH} ${EC_LD_LIBRARY_PATH}) \
		$(s.prefix "-l" ${LIBRARIES} ) \
		"$@"
else
   if [[ -n $Verbose ]] ; then
      cat <<EOF
${CC:-ERROR_CC_undefined} ${SourceFile} $CC_options_NOLD ${CFLAGS} \\
		$(s.prefix "" ${DEFINES} ) \\
		$(s.prefix "-I" ${INCLUDES} ${EC_INCLUDE_PATH}) \\
		"$@"
EOF
   fi
	${CC:-ERROR_CC_undefined} ${SourceFile} $CC_options_NOLD ${CFLAGS} \
		$(s.prefix "" ${DEFINES} ) \
		$(s.prefix "-I" ${INCLUDES} ${EC_INCLUDE_PATH}) \
		"$@"
fi
