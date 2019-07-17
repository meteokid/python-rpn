#!/usr/bin/ksh
set -ex

rm -rf *.mod core *.o arjen.db toto titi *.a \
       read1_* obs_* elements_* readburp_* \
       write1_* write2_* write2f_* write2d_* readfloat_* test_* ii_files \
       alain_* obs_* readfloat_*

# load appropriate compilers for each architecture
if [[ -z ${COMP_ARCH} ]]; then
    if [[ "${ORDENV_PLAT}" = "aix-7.1-ppc7-64" ]]; then
        . ssmuse-sh -d hpcs/201402/01/base -d hpcs/ext/xlf_13.1.0.10
    elif [[ "${ORDENV_PLAT}" = "ubuntu-10.04-amd64-64" || "${ORDENV_PLAT}" = "ubuntu-12.04-amd64-64" ]]; then
        . ssmuse-sh -d hpcs/201402/01/base -d hpcs/201402/01/intel13sp1u2
    elif [[ "${ORDENV_PLAT}" = "ubuntu-14.04-amd64-64" ]]; then
        . r.load.dot  /ssm/net/hpcs/201402/01/base /ssm/net/hpcs/exp/intel2016/01
    else
       echo "Unsupported architecture: ${ORDENV_PLAT}"
       exit 1
    fi
fi
. ssmuse-sh -d rpn/libs/16.2.1

if [[ "${ORDENV_PLAT}" = "aix-7.1-ppc7-64" ]]; then
    platform_parameters="-libsys C"
elif [[ "${ORDENV_PLAT}" = "ubuntu-10.04-amd64-64" || "${ORDENV_PLAT}" = "ubuntu-12.04-amd64-64" ]]; then
    if [[ "${COMP_ARCH}" = "intel13sp1u2" ]]; then
        platform_parameters="-libsys stdc++"
    elif [[ "${COMP_ARCH}" = "pgi1401" ]]; then
        platform_parameters="-libsys std C stdc++ numa"
    fi
elif [[ "${ORDENV_PLAT}" = "ubuntu-14.04-amd64-64" ]]; then
    platform_parameters="-libsys stdc++"
fi

# alain.cpp: on AIX produces the obscure message: 1586-494 (U) INTERNAL COMPILER ERROR: Wcode stack is not empty at beginning of basic block.
# test.cpp : same message as above on AIX
# write2d.cpp : *** FATAL ERROR #44 from module burp_valid789: invalid code for datyp 7
set -A files read1.c readcc.cpp readburp.c readfloat.c setit.cpp write1.cpp write2.cpp write2f.cpp maxlen.cpp obs.cpp val.cpp elements.cpp
for file in ${files[@]}; do
    echo $file
    binary=`echo $file | cut -f1 -d"."`
    s.compile -o $binary -src $file -includes ../include -libpath ../lib -libappl burp_c -bidon c -main my_main -librmn rmn_016 $platform_parameters -debug -O 3
done

