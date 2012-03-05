#!/bin/ksh

[[ -e gem_settings.nml && ! -L gem_settings.nml ]] && mv gem_settings.nml gem_settings.nml_bk_$$

for item in LAM GLB ; do

	 rm -f gem_settings.nml ; ln -s tests/gem_settings.nml_$item gem_settings.nml
	 runent > run_${item}.log 2>&1
	 runmod >> run_${item}.log 2>&1

done
