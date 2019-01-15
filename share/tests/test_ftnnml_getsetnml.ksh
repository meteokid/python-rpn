#!/bin/ksh

nmlfile=$TMPDIR/mynml$$

cat > $nmlfile <<EOF
&Abc
  Qwe = 1
/
EOF

my_assert_equal() {
   [[ "$1" != "$2" ]] && echo "FAILED: $3 ; got '$1' expected '$2'" 1>&2 || true
   [[ "$1" == "$2" ]] && echo "OK: $3" || true
}

toto="$(rpy.nml_get -f $nmlfile abc/qwe)"
my_assert_equal "$toto" "1" "rpy.nml_get case insensitive"

rpy.nml_set -f $nmlfile  abc/Asd=3
toto="$(rpy.nml_get -f $nmlfile abc/asd)"
my_assert_equal "$toto" "3" "rpy.nml_set case insensitive"

rm -f $nmlfile
