#! /usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ
import sys
from bh import bhlib, actions

def _init(b):
   
   environ["BH_PROJECT_NAME"] = "vgriddescriptors"
   environ["BH_PACKAGE_NAME"]  = "%(BH_PROJECT_NAME)s" % environ
   environ["BH_PACKAGE_NAMES"] = "%(BH_PROJECT_NAME)s" % environ
   environ["BH_PACKAGE_CONTROL_DIR"] = "%(BH_HERE_DIR)s" % environ

def _make(b):
   
    b.shell("""
            set -e
            cd ${BH_PULL_SOURCE}
            REMOTE_NAME=gitlab_com
            REMOTE=$(git remote -v | grep fetch | grep ${REMOTE_NAME} | awk '{print $2}')
            if [ "${REMOTE}" = "" ];then
               echo "ERROR git remote ${REMOTE_NAME} not found"
               exit 1
            fi
            (
             CONTROL_DIR=${BH_PACKAGE_CONTROL_DIR}/${BH_PROJECT_NAME}/.ssm.d
             mkdir -p ${CONTROL_DIR}
             cp ${BH_TOP_BUILD_DIR}/post-install ${CONTROL_DIR}
             CONTROL_FILE=${CONTROL_DIR}/control.template
             echo \"Package: ${BH_PACKAGE_NAME}\"                                                                  > ${CONTROL_FILE}
             echo \"Version: ${BH_PULL_SOURCE_GIT_BRANCH}\"                                                       >> ${CONTROL_FILE}
             echo \"Platform: ${ORDENV_PLAT}\"                                                                    >> ${CONTROL_FILE}
             echo \"Maintainer: cmdn (A. Plante)\"                                                                >> ${CONTROL_FILE}
             echo \"BuildInfo: git clone ${REMOTE}\"                                                              >> ${CONTROL_FILE}
             echo \"           cd in new directory created\"                                                      >> ${CONTROL_FILE}
             echo \"           git checkout -b temp ${BH_PULL_SOURCE_GIT_BRANCH}\"                                >> ${CONTROL_FILE}
             echo \"           # or git checkout -b temp $(git rev-parse HEAD)\"                                  >> ${CONTROL_FILE}
             echo \"           cd src\"                                                                           >> ${CONTROL_FILE}
             echo \"           . setup.dot\"                                                                      >> ${CONTROL_FILE}
             echo \"           make\"                                                                             >> ${CONTROL_FILE}
             echo \"Vertical grid descriptors package\"                                                           >> ${CONTROL_FILE}
             cd ${BH_BUILD_DIR}/src
             make vgrid_version
             make shared
            )""",environ)
   
def _test(b):
    b.shell("""
        (
         set -e
         cd ${BH_BUILD_DIR}/tests
         make tests
        )""",environ)

def _install(b):
    b.shell("""
        (
         set -e        
         mkdir -p ${BH_INSTALL_DIR}/lib
         cd ${BH_INSTALL_DIR}/lib
         cp ${BH_TOP_BUILD_DIR}/src/libdescrip.a  libdescrip_${BH_PULL_SOURCE_GIT_BRANCH}.a
         ln -s libdescrip_${BH_PULL_SOURCE_GIT_BRANCH}.a libdescrip.a
         if [ -f ${BH_TOP_BUILD_DIR}/src/libdescripshared.so ];then
            cp ${BH_TOP_BUILD_DIR}/src/libdescripshared.so libdescripshared_${BH_PULL_SOURCE_GIT_BRANCH}.so
            ln -s libdescripshared_${BH_PULL_SOURCE_GIT_BRANCH}.so libdescripshared.so
         fi
         mkdir -p ${BH_INSTALL_DIR}/include
         cd ${BH_INSTALL_DIR}/include
         cp ${BH_TOP_BUILD_DIR}/src/*.mod .
         cp ${BH_TOP_BUILD_DIR}/src/*.h .
         cp ${BH_TOP_BUILD_DIR}/src/vgrid_version.h* .
         mkdir -p ${BH_INSTALL_DIR}/src
         cd ${BH_TOP_BUILD_DIR}/src
         make clean
         cp * ${BH_INSTALL_DIR}/src
        )""")

if __name__ == "__main__":
   dr, b = bhlib.init(sys.argv, bhlib.PackageBuilder)
   b.actions.set("init", _init)
   b.actions.set("pull", [actions.pull.git_archive])
   b.actions.set("clean", ["""(cd ${BH_BUILD_DIR}/src; make clean)"""])
   b.actions.set("make", _make)
   #b.actions.set("test",_test)
   b.actions.set("install", _install)
   b.actions.set("package", actions.package.to_ssm)

   b.supported_platforms = [
      "ubuntu-12.04-amd64-64",
      "ubuntu-14.04-amd64-64",
      "aix-7.1-ppc7-64",
   ]
   dr.run(b)

   b.supported_modes = [
      "intel",
      "xlf13",
   ]

# Exemple d'appel:
#   ./bh_vgrid.py -p ubuntu-12.04-amd64-64 -m intel
