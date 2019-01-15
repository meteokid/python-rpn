#!/bin/ksh
#
printf "\n=====>  Um_output_lastmv.ksh starts: `date` ###########\n\n"

DIR_dst=$1
last=$2

cd ${DIR_dst}/$last

if [ $? -eq 0 ] ; then

  Um_checkfs.ksh `pwd`
  if [ $? -ne 0 ] ; then exit 1 ; fi

  for i in * ; do
    if [ -d $i ] ; then
      mkdir -p ${DIR_dst}/$i
      printf "DIR $i mv $i/* ${DIR_dst}/$i\n"
      mv $i/* ${DIR_dst}/$i
    else
      printf "FILE $i mv $i ${DIR_dst}\n"
      mv $i ${DIR_dst}
    fi
  done

  cd ${DIR_dst}

  /bin/rm -rf $last

fi
#
printf "\nUm_output_lastmv.ksh ends: `date`\n\n"
