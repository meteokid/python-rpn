#!/bin/ksh


export __version=GEM/x/4.4.0-b11-run
__machlist="arxt20:Linux_x86-64 zeta:AIX spica:AIX-powerpc7"
__gembndl="gem_${__version##*/}-s.bndl"
__pathlist="PATH MANPATH LD_LIBRARY_PATH PYTHONPATH TCL_LIBRARY LIBPATH EC_INCLUDE_PATH EC_LD_LIBRARY_PATH SSM_INCLUDE_PATH NLSPATH"
__excludelist="${__pathlist} PWD _ TMPDIR BATCH_TMPDIR BIG_TMPDIR ZZZPID"


do_one_mach() {
	 __mach=$1
	 __basearch=$2
	 __tmpdir=~/tmp
	 [[ ! -d ~/tmp ]] && __tmpdir=~

	 __log=${__tmpdir}/gemenv.${__version##*/}.${__basearch}.$$.log
	 __getenv=${__tmpdir}/getgemenv.${__version##*/}.${__basearch}.$$.sh
	 __gemuse=gemenv.${__version##*/}-s.${__basearch}.sh

	 cat > ${__getenv} <<EOF
#!/bin/ksh
    . ~/.profile >/dev/null 2>&1
	 env|sort > ${__log}.0
	 #env|sort|cut -d= -f1 > ${__log}.0.nm
    for item in ${__pathlist} ; do
	    echo \$(eval echo \\\$\$item) | tr ' ' ':' | tr ':' '\n' > ${__log}.0.\${item}
    done

	 . s.ssmuse.dot ${__version} > /dev/null 2>&1 
	 env|sort > ${__log}.1
	 #env|sort|cut -d= -f1 > ${__log}.1.nm
    for item in ${__pathlist} ; do
	    echo \$(eval echo \\\$\$item) | tr ' ' ':' | tr ':' '\n' > ${__log}.1.\${item}
    done
EOF
	 chmod u+x ${__getenv}
	 ssh ${__mach} ${__getenv} >/dev/null 2>&1

	 echo "\necho ${__version} UNIX shell Environment setup" > ${__gemuse}

	 #export __mylist="$(diff -ibw ${__log}.0.nm ${__log}.1.nm | grep '>' | cut -d' ' -f2)"
	 export __mylist="$(diff -ibw ${__log}.0 ${__log}.1 | grep '>' | cut -d' ' -f2 | cut -d= -f1)"
	 (
		  #eval "$(cat ${__log}.1 | sed -r 's/=(.*)$/=\"\1\"/' | grep -v '^PWD=' | grep -v '^_=' | grep -v 'TMPDIR=' | grep -v 'PID=')";
		  for item in ${__mylist} ; do
				__toinclude=1
				for item2 in ${__excludelist} ; do
					 [[ x$item2 == x$item ]] && __toinclude=0
				done
				__sss=''
				[[ x$item == xATM_MODEL_BNDL ]] && __sss=-s
				#[[ x$__toinclude == x1 ]] && echo export ${item}=\"$(eval echo \$$item)${__sss}\" >> ${__gemuse}
				[[ x$__toinclude == x1 ]] && echo export $(cat ${__log}.1 | grep "^${item}" | sed -r "s/=(.*)\$/=\"\1${__sss}\"/") >> ${__gemuse}
		  done ;
	 )

	 for item in ${__pathlist} ; do
		  __pathparts="$(diff ${__log}.0.${item} ${__log}.1.${item}| grep '>' | cut -d' ' -f2 | tr '\n' ':')"
		  __sep=:
		  if [[ "x${__pathparts}" != x ]] ; then
				[[ x${item%%_*} == xEC ]] && __pathparts="$(echo ${__pathparts%:} | tr ':' ' ')" && __sep=' '
				cat >> ${__gemuse} <<EOF
export ${item}="${__pathparts%:}${__sep}\$${item}"
EOF
		  fi
		  rm -f ${__log}.0.${item} ${__log}.1.${item}
	 done
	 cat >> ${__gemuse} <<EOF
eval \$(s.path_cleanup ${__pathlist})
export MANPATH="\${MANPATH%:}:"
EOF

	 cat >> ${__gembndl} <<EOF
 ${__basearch}:GEM/others/smdot/${__gemuse%.*}
EOF

	 #rm -f ${__getenv} ${__log}.0 ${__log}.1 ${__log}.0.nm ${__log}.1.nm
	 rm -f ${__getenv}
}

#==== ====
__bndlnotfound="$(s.resolve_ssm_shortcuts ${__version} 2>&1 | grep ${__version})"

if [[ x"${__bndlnotfound}" != x ]] ; then
	 cat <<EOF
ERROR: SSM shortcut ${__version} cannot be found
=== ABORT ===
EOF
	 exit 1
fi

mv ~/.profile.d ~/.profile.d-$$

echo "Prepend GEM/others/gem-prep-1" > ${__gembndl}
for __mymach in $__machlist ; do
	 do_one_mach ${__mymach%%:*} ${__mymach##*:} 
done

rm -rf ~/.profile.d
mv ~/.profile.d-$$ ~/.profile.d

cat >> ${__gembndl} <<EOF
 GEM/others/gem-post-1
EOF
mv ${__gembndl} ${__gembndl}-$$
cat ${__gembndl}-$$ | tr '\n' ' ' > ${__gembndl}
echo >> ${__gembndl}
rm -f ${__gembndl}-$$

cat<<EOF
WARNING: Additional Installation steps
  chmod u+w ~armnenv/SsmBundles/${__version%/*}
  mv ${__gembndl} ~armnenv/SsmBundles/${__version}.bndl
  chmod 444 ~armnenv/SsmBundles/${__version}.bndl
  chmod u-w ~armnenv/SsmBundles/${__version%/*}

  chmod u+w ~armnenv/SsmBundles/GEM/others/smdot
  mv gemenv.${__version##*/}-s.*.sh ~armnenv/SsmBundles/GEM/others/smdot/
  chmod 444 ~armnenv/SsmBundles/GEM/others/smdot/gemenv.${__version##*/}-s.*.sh
  chmod u-w ~armnenv/SsmBundles/GEM/others/smdot
EOF
