#!/bin/ksh
#
arguments=$*
eval `cclargs_lite -D " " $0 \
  -cfg         "cfg_0000"          "cfg_0000"         "[Analysis file or archive    ]"\
  -gemnml      "gem_settings.nml"  "gem_settings.nml" "[Model namelist settings file]"\
  -nmlfile     "checkdm.nml"       "checkdm.nml"      "[# of simultaneous threads   ]"\
  ++ $arguments`

set -ex

BIN=$(which checkdmpart_${BASE_ARCH}.Abs)

ici=${PWD}
ROOT_WORK=${PWD}/checkdmpart$$
WORKDIR=${ROOT_WORK}/${cfg}
/bin/rm -rf ${ROOT_WORK} ; mkdir -p ${WORKDIR}
cp ${gemnml}  ${WORKDIR}/model_settings.nml
cp ${nmlfile} ${ROOT_WORK}
domain=$(echo ${cfg##*_} | sed 's/^0*//')
if [ -z "${domain}" ] ; then domain=0 ;fi

cd ${ROOT_WORK}

unset GEM_YINYANG
GRDTYP=$(getnml -f ${WORKDIR}/model_settings.nml -n grid grd_typ_s 2> /dev/null | sed "s/'//g")
if [ "$GRDTYP" == "GY" ] ; then 
  GEM_YINYANG=YES
  mkdir -p ${WORKDIR}/YIN/000-000 ${WORKDIR}/YAN/000-000
  ln -s $AFSISIO/datafiles/constants/thermoconsts ${WORKDIR}/YIN/000-000/constantes
  ln -s $AFSISIO/datafiles/constants/thermoconsts ${WORKDIR}/YAN/000-000/constantes
  cp checkdm.nml ${WORKDIR}/YIN/000-000
  mv checkdm.nml ${WORKDIR}/YAN/000-000
  ngrids=2
else
  mkdir -p ${WORKDIR}/000-000
  ln -s $AFSISIO/datafiles/constants/thermoconsts ${WORKDIR}/000-000/constantes
  mv checkdm.nml ${WORKDIR}/000-000
  ngrids=1
fi

export GEM_YINYANG
export DOMAIN_start=$domain
export DOMAIN_end=$domain
export DOMAIN_total=1
export GEM_NDOMAINS=$domain:$domain
export TASK_INPUT=${PWD}
export TASK_WORK=${PWD}

printf "\n RUNNING ${BIN} \n\n"
r.run_in_parallel -pgm $BIN -npex ${ngrids} -inorder

mv checkdmpart_status.dot ${ici} || true
cd $ici
/bin/rm -rf ${ROOT_WORK} || true


