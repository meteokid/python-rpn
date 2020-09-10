#!/bin/bash


my_assert_equal() {
   [[ "$1" != "$2" ]] && echo "FAILED: $3 ; got '$1' expected '$2'" 1>&2 || true
   [[ "$1" == "$2" ]] && echo "OK: $3" || true
}

tmpoutfile1=${TMPDIR:-/tmp}/rpytest.cclargs.out1.$$
tmpoutfile2=${TMPDIR:-/tmp}/rpytest.cclargs.out2.$$
bin1cclargs=${TMPDIR:-/tmp}/rpytest.cclargs.$$
cat > ${bin1cclargs} <<EOF
    eval \$(rpy.cclargs \
           -D ":" \
           --desc       "Script description" \
           --epilog     "Help Epilog message" \
           --positional "Positional args description" \
           \${0##*/} \
           "-opt1" "val1"   "val2" "[desc]" \
           "-opt2" "=-val1" "val2" "[desc]" \
           ++ \$*)
    echo positional="\$*"
    echo opt1="\$opt1"
    echo opt2="\$opt2"
EOF
chmod a+x ${bin1cclargs}

bin2cclargparse=${TMPDIR:-/tmp}/rpytest.cclargparse.$$
cat > ${bin2cclargparse} <<EOF
    eval \$(rpy.cclargparse \
           -D ":" \
           \${0##*/} \
           "Script description" \
           "Help Epilog message" \
           "positional"   'nargs=2'        'default_pos'  '[description_pos]' \
           "--opt1"       'required=True'  'default_op1'  '[description_op1]' \
           "--opt2"       'nargs=*'        'default_op2'  '[description_op2]' \
           "-v,--verbose" 'action=count'          '0'     '[description_v]' \
           "+v"           'action=count, dest=v2' '2'     '[description_v2]' \
           "--keep,-k"    'action=store_true'     'false' '[description_k]' \
           "--num,-n"     'type=int'              '2'     '[description_num]' \
           ++++ \$*)
    echo positional="\$*"
    echo opt1="\$opt1"
    echo opt2="\$opt2"
    echo verbose="\$verbose"
    echo v2="\$v2"
    echo keep="\$keep"
    echo num="\$num"
EOF
chmod a+x ${bin2cclargparse}

#==== Help ====
${bin1cclargs} -h > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargs status - help"
cat > ${tmpoutfile2} <<EOF
usage: ${bin1cclargs##*/} [-h] [positional] [options]

Script description

positional arguments:
  positional        Positional args description

optional arguments:
  -h, --help        Print this help/usage message
  --opt1 [ [ ...]]  desc ["val1"|"val2"]
  --opt2 [ [ ...]]  desc ["-val1"|"val2"]

Help Epilog message
EOF
my_assert_equal "$(cat ${tmpoutfile2})" "$(cat ${tmpoutfile1})" "cclargs - help"

${bin2cclargparse} -h > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - help"
cat > ${tmpoutfile2} <<EOF
usage: ${bin2cclargparse##*/} [-h] [positional] [options]

Script description

positional arguments:
  positional            description_pos (default='default_pos') (nargs=2)

optional arguments:
  -h, --help            Print this help/usage message
  --opt1 OPT1           description_op1 (default='default_op1')
  --opt2 [OPT2 [OPT2 ...]]
                        description_op2 (default='default_op2') (nargs=*)
  -v, --verbose         description_v (default=0)
  +v                    description_v2 (default=0)
  --keep, -k            description_k (default=False)
  --num NUM, -n NUM     description_num (default=2)

Help Epilog message
EOF
my_assert_equal "$(cat ${tmpoutfile2})" "$(cat ${tmpoutfile1})" "cclargparse - help"

#==== Errors ====

${bin1cclargs} -nosucharg > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargs status - nosucharg"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "ERROR: (${bin1cclargs##*/}) unrecognized arguments: --nosucharg;" \
                "cclargs - nosucharg"

${bin2cclargparse} --opt1 opt1val > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - missing positional"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                ";ERROR: (${bin2cclargparse##*/}) the following arguments are required: positional;" \
                "cclargparse - missing positional"

${bin2cclargparse} p1 p2 --opt2 opt1val > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - missing required opt1"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                ";ERROR: (${bin2cclargparse##*/}) the following arguments are required: --opt1;" \
                "cclargparse - missing required opt1"

${bin2cclargparse} p1 p2 p3 --opt1 opt1val > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - too many positional"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "ERROR: (${bin2cclargparse##*/}) unrecognized arguments: p3;" \
                "cclargparse - too many positional"

${bin2cclargparse} p1 p2 --opt1 opt1val --nosucharg > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - nosucharg"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "ERROR: (${bin2cclargparse##*/}) unrecognized arguments: --nosucharg;" \
                "cclargparse - nosucharg"

${bin2cclargparse} p1 p2 --opt1 > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - missing mandatory value"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                ";ERROR: (${bin2cclargparse##*/}) argument --opt1: expected one argument;" \
                "cclargparse - missing mandatory value"

${bin2cclargparse} p1 p2 --opt1 opt1val -n abc > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "1" "cclargparse status - bad value type"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                ";ERROR: (${bin2cclargparse##*/}) argument --num/-n: invalid int value: 'abc';" \
                "cclargparse - bad value type"

#==== Ok values ====

${bin1cclargs} -opt1 > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargs status - opt1"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=;opt1=val2;opt2=-val1;" \
                "cclargs - opt1"

${bin1cclargs} -opt1 opt1val > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargs status - opt1 val"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=;opt1=opt1val;opt2=-val1;" \
                "cclargs - opt1 val"

${bin1cclargs} -opt1 opt1val1 opt1val2 > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargs status - opt1 vals"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=;opt1=opt1val1 opt1val2;opt2=-val1;" \
                "cclargs - opt1 vals"

${bin1cclargs} a b -opt1 opt1val1 > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargs status - opt1 val + positional"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=a b;opt1=opt1val1;opt2=-val1;" \
                "cclargs - opt1 val + positional"

${bin1cclargs} a b  > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargs status - positional"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=a b;opt1=val1;opt2=-val1;" \
                "cclargs - positional"


${bin2cclargparse} p1 p2 --opt1 opt1val -v -v -k --num=3> ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargparse status - opt1 val"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=p1:p2;opt1=opt1val;opt2=default_op2;verbose=2;v2=0;keep=True;num=3;" \
                "cclargparse - opt1 val"

${bin2cclargparse} p1 p2 --opt1 opt1val --opt2 optval1 optval2 > ${tmpoutfile1} 2>&1
my_assert_equal "$?" "0" "cclargparse status - opt2 vals"
my_assert_equal "$(cat ${tmpoutfile1} | tr '\n' ';')" \
                "positional=p1:p2;opt1=opt1val;opt2=optval1:optval2;verbose=0;v2=0;keep=False;num=2;" \
                "cclargparse - opt2 vals"

rm -f ${bin1cclargs} ${bin2cclargparse} ${tmpoutfile1} ${tmpoutfile2}
