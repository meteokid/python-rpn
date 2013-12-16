MAKE   = make
PYARCH = linux-x86_64-2.6
CCNAME = intel
CCOPT  = -Wtrigraphs -fpic
LDFLAGS=-static-intel

# $CC ${SourceFile} $CC_options ${CFLAGS} \
# 	$(s.prefix "" ${DEFINES} ) \
# 	$(s.prefix "-I" ${INCLUDES} ${EC_INCLUDE_PATH}) \
# 	$(s.prefix "-L" ${LIBRARIES_PATH} ${EC_LD_LIBRARY_PATH}) \
# 	$(s.prefix "-l" ${LIBRARIES} ) \
# 	"$@"
# $CC ${SourceFile} ${CC_options_NOLD:-${CC_options}} ${CFLAGS} \
# 	$(s.prefix "" ${DEFINES} ) \
# 	$(s.prefix "-I" ${INCLUDES} ${EC_INCLUDE_PATH}) \
# 	"$@"

# icc -Wtrigraphs -fpic -I. -c Fstdc.c -DWITHOUT_OpenMP -I/ssm/net/rpn/libs/201309/01/ubuntu-10.04-amd64-64/include -I/ssm/net/rpn/libs/201309/01/ubuntu-10.04-amd64-64/include/Linux_x86-64/intel13sp1 -I/ssm/net/hpcs/201311/00-test/intel13sp1/ubuntu-10.04-amd64-64/include -I/ssm/net/hpcs/201311/00-test/intel13sp1/ubuntu-10.04-amd64-64/include/Linux_x86-64/intel13sp1 -I/ssm/net/hpcs/201311/00-test/intel13sp1/all/include -I/ssm/net/hpcs/201311/00-test/intel13sp1/all/include/Linux_x86-64/intel13sp1 -I/ssm/net/hpcs/201311/00-test/intel13sp1/multi/include -I/ssm/net/hpcs/201311/00-test/intel13sp1/multi/include/Linux_x86-64/intel13sp1 -I/ssm/net/hpcs/201311/00-test/base/ubuntu-10.04-amd64-64/include -I/ssm/net/hpcs/201311/00-test/base/ubuntu-10.04-amd64-64/include/Linux_x86-64/intel13sp1 -I/ssm/net/hpcs/201311/00-test/base/all/include -I/ssm/net/hpcs/201311/00-test/base/all/include/Linux_x86-64/intel13sp1 -I/ssm/net/isst/maestro/1.3.0/linux26-x86-64/include -I/ssm/net/isst/maestro/1.3.0/linux26-x86-64/include/Linux_x86-64/intel13sp1 -I/ssm/net/isst/maestro/1.3.0/linux26-i686/include -I/ssm/net/isst/maestro/1.3.0/linux26-i686/include/Linux_x86-64/intel13sp1 -I/ssm/net/isst/maestro/1.3.0/all/include -I/ssm/net/isst/maestro/1.3.0/all/include/Linux_x86-64/intel13sp1 -I/users/dor/armn/sch/ECssm/multi/include -I/users/dor/armn/sch/ECssm/multi/include/Linux_x86-64/intel13sp1 -I/users/dor/armn/sch/ECssm/all/include -I/users/dor/armn/sch/ECssm/all/include/Linux_x86-64/intel13sp1 -I/home/ordenv/ssm-domains/ssm-base/all/include -I/home/ordenv/ssm-domains/ssm-base/all/include/Linux_x86-64/intel13sp1 -I/ssm/net/sss/base/20131115/all/include -I/ssm/net/sss/base/20131115/all/include/Linux_x86-64/intel13sp1 -I/ssm/net/hpcs/201311/00-test/intel13sp1/ubuntu-10.04-amd64-64/include -I/ssm/net/hpcs/201311/00-test/intel13sp1/all/include -I/ssm/net/hpcs/201311/00-test/intel13sp1/multi/include -I/ssm/net/hpcs/201311/00-test/base/ubuntu-10.04-amd64-64/include -I/ssm/net/hpcs/201311/00-test/base/all/include -I/ssm/net/isst/maestro/1.3.0/linux26-x86-64/include -I/ssm/net/isst/maestro/1.3.0/linux26-i686/include -I/ssm/net/isst/maestro/1.3.0/all/include -I/users/dor/armn/sch/ECssm/multi/include -I/users/dor/armn/sch/ECssm/all/include -I/home/ordenv/ssm-domains/ssm-base/all/include -I/ssm/net/sss/base/20131115/all/include
