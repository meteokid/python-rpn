  cat <<__EOF
  Error in $(basename $0)
  $*
  --- ABORT ---
__EOF
  . r.return.dot
  exit 1
