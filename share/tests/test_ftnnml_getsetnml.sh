#!/bin/bash

nmlfile=${TMPDIR}/mynml$$

cat > ${nmlfile} <<EOF
&Abc
  Qwe = 1
/

&Abcd
  Qwes = 'zxs'
  Qwed = "zxd"
/
EOF

my_assert_equal() {
   [[ "$1" != "$2" ]] && echo "FAILED: $3 ; got '$1' expected '$2'" 1>&2 || true
   [[ "$1" == "$2" ]] && echo "OK: $3" || true
}

toto="$(rpy.nml_get -f ${nmlfile} abc/abs 2>/dev/null)"
my_assert_equal "${toto}" "" "rpy.nml_get not found var"

toto="$(rpy.nml_get -f ${nmlfile} cba/qwe 2>/dev/null)"
my_assert_equal "${toto}" "" "rpy.nml_get not found nml"


toto="$(rpy.nml_get -f ${nmlfile} abc/qwe)"
my_assert_equal "${toto}" "1" "rpy.nml_get case insensitive"

toto="$(rpy.nml_get -f ${nmlfile} abcd/qwes)"
my_assert_equal "${toto}" "'zxs'" "rpy.nml_get with s-quotes"

toto="$(rpy.nml_get -f ${nmlfile} abcd/qwed)"
my_assert_equal "${toto}" '"zxd"' "rpy.nml_get with d-quotes"

toto="$(rpy.nml_get -f ${nmlfile} --unquote abcd/qwes)"
my_assert_equal "${toto}" "zxs" "rpy.nml_get without s-quotes"

toto="$(rpy.nml_get -f ${nmlfile} --unquote  abcd/qwed)"
my_assert_equal "${toto}" "zxd" "rpy.nml_get without d-quotes"

toto="$(rpy.nml_get -f ${nmlfile} --keys abc/qwe)"
my_assert_equal "${toto}" "qwe=1" "rpy.nml_get with key"

toto="$(rpy.nml_get -f ${nmlfile} --listnml)"
my_assert_equal "${toto}" "abc abcd" "rpy.nml_get list namelist"

toto="$(rpy.nml_get -f ${nmlfile} --listkeys)"
my_assert_equal "${toto}" "qwe qwes qwed" "rpy.nml_get list keys"

toto="$(rpy.nml_get -f ${nmlfile} --listkeyval | sort | tr '\n' ';')"
my_assert_equal "${toto}" "abc/qwe=1;abcd/qwed=\"zxd\";abcd/qwes='zxs';" "rpy.nml_get list keyval"

toto="$(rpy.nml_get -f ${nmlfile} --prettyprint)"
my_assert_equal "${toto}" "$(cat ${nmlfile} | tr 'A-Z' 'a-z' | sed 's/ //g')" "rpy.nml_get prettyprint"

newval=abc/qwe=3
rpy.nml_set -f ${nmlfile} ${newval}
toto="$(rpy.nml_get -f ${nmlfile} ${newval%=*})"
my_assert_equal "${toto}" "${newval#*=}" "rpy.nml_set new val"

newval=abc/abs=4
rpy.nml_set -f ${nmlfile} ${newval}
toto="$(rpy.nml_get -f ${nmlfile} ${newval%=*})"
my_assert_equal "${toto}" "${newval#*=}" "rpy.nml_set new var"

newval=cba/wer=5
rpy.nml_set -f ${nmlfile} ${newval}
toto="$(rpy.nml_get -f ${nmlfile} ${newval%=*})"
my_assert_equal "${toto}" "${newval#*=}" "rpy.nml_set new nml"


getval=abc/nosuchvar
rpy.nml_del -f ${nmlfile} ${getval} 2>/dev/null
toto="$(rpy.nml_get -f ${nmlfile} ${getval} 2>/dev/null)"
my_assert_equal "${toto}" "" "rpy.nml_del no such var"

getval=abc/qwe
rpy.nml_del -f ${nmlfile} ${getval}
toto="$(rpy.nml_get -f ${nmlfile} ${getval} 2>/dev/null)"
my_assert_equal "${toto}" "" "rpy.nml_del var"

#TODO: rpy.nml_get -f ${nmlfile} --clean
#TODO: rpy.nml_get -f ${nmlfile} --downcase

filename0=${rpnpy:-.}/share/tests/data/gem_settings.nml.ref0
filename1=${rpnpy:-.}/share/tests/data/gem_settings.nml.ref1
cp ${filename0} ${nmlfile}
rpy.nml_clean -f ${nmlfile} -d -c -s -m 300 -t 'xst_stn_latlon(lat,lon,name) xst_stn_ij(i,j,name)' >/dev/null
toto="$(diff ${filename1} ${nmlfile})"
my_assert_equal "${toto}" "" "rpy.nml_clean"

rm -f ${nmlfile}
