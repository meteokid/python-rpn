#!/bin/ksh
#
arguments=$*
eval `cclargs_lite $0 \
  -src       ""	  ""         "[]"\
  -dst_mach   ""	  ""         "[]"\
  -dst_dir   ""	  ""         "[]"\
  -abortf   "Um_output" "Um_output" "[abort file     ]"\
  ++ $arguments`
#  -domain    ""	  ""         "[]"\

printf "\n=====>  Um_download.ksh starts: `date` ###########\n\n"

abort_file=${TASK_WORK}/${abortf}_$$
touch ${abort_file}
set -ex

#  CHECKSUM=$(which Um_checksum.ksh)
#  src_md5=source_md5_${DOMAIN}_$(basename ${src_file}).lis
#  dst_md5=destination_md5_${DOMAIN}_$(basename ${src_file}).lis
#  ssh $src_mach "${CHECKSUM} ${src_file}" > ${src_md5}

#SRC=$(r.read_link ${src})
SRC=$(readlink ${src})
dst=${dst_dir}/$(basename ${src})

if [ "${dst_mach}" == "${TRUE_HOST}" ] ; then
  /bin/rm -f ${dst}
  ln -s ${SRC} ${dst}
else
  if [ "$dst_mach" == "eccc-ppp1" -o "$dst_mach" == "eccc-ppp2" ] ; then
    /bin/rm -rf ${dst}
    cp -r ${SRC} ${dst}
  else
    sscp -r ${SRC} ${dst_mach}:${dst}
  fi
fi

#  Um_checksum.ksh ${TASK_WORK}/${DOMAIN}/last* > ${dst_md5}
#  cnt=$(diff ${src_md5} ${dst_md5} | wc -l)
#  if [ ${cnt} -gt 0 ] ; then
#     printf "\n Problem with data transfer from ${src_mach} - ABORT\n\n"
#     exit 1
#  fi
#  rep_in=${TASK_WORK}/${DOMAIN}/last*

/bin/rm -f ${abort_file}

printf "\n=====>  Um_download.ksh ends: `date` ###########\n\n"



