#!/bin/ksh

__bundlename=$1

cat <<EOF
============================================================================
WARNING: You are using an experimental verion of a software ($__bundlename)
         Please note that 
         - It may contains serious bugs
         - It will only be avail. for a limited time (until next version)
----------------------------------------------------------------------------
ATTENTION: Vous utilisez un version experimentale d'un logiciel ($__bundlename)
           SVP noter que:
           - Il peut contenir des defectuosites serieuses
           - Il ne sera offert que pour une periode limite 
             (jusqu'a la sortie de la prochaine version)
============================================================================
EOF
