#!/bin/ksh
#TODO: KEEP OR NOT? not used in model - only in Sortie.cfg

#====> Obtaining the arguments:
eval `cclargs_lite $0 \
     -list    ""  ""    "[list of files                      ]"\
     -src     ""  ""    "[Source directory                   ]"\
     -dst     ""  ""    "[Destination directory              ]"\
     -dplusp  "0" "1"   "[Combine dynamics and physics output]"\
  ++ $*`
#
set -ex

EDIT=editfst+

for i in $list ; do
  fn=`basename $i`
  dyn=`echo $fn | grep ^d | wc -l`
  phy=`echo $fn | grep ^p | wc -l`
  if [ $dplusp -gt 0 ] ; then
    destination=`echo $fn | sed 's/\(^.\)\(.*\)/\2/'`
    if [ $dyn -gt 0 ] ; then
      cp ${src}/$i ${dst}/${destination}
    else
      $EDIT -e -s ${src}/${i} -d ${dst}/${destination} -i /dev/null
    fi
  else
    destination=$fn
    ln -s $(true_path ${src})/$i ${dst}
  fi
done
#
exit 0
