#!/usr/bin/ksh
set -ex

SCRIPT=`readlink -f $0`
SCRIPT_PATH=`dirname $SCRIPT`
cd $TMPDIR

directory=/data/ade/banco/postalt/r1
today=`date +%Y%m%d`00_

rm -rf *_
scp joule:$directory/$today .

# load cmoi domain for AFSISIO environment variable (required by rmnlib for qrbsct function)
. ssmuse-sh -d /ssm/net/cmoi/base/20130927

# readburp and readfloat produce too much output for testing so have been removed from list
set -A files read1 setit maxlen obs readcc write2 write2f
for file in ${files[@]}; do
    echo $file
    read "Enter"
    $SCRIPT_PATH/$file ./$today
done

set -A files elements val
for file in ${files[@]}; do
    echo $file
    read "Enter"
    $SCRIPT_PATH/$file -f ./$today
done

echo "write1"
read "Enter"
$SCRIPT_PATH/write1 ./$today ./junk_output

# cleanup temporary files that were created by the above programs
rm -rf *_ file_write1 file_write1f ./junk_output

