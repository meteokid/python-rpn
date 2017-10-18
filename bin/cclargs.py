#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Parse list of args for bash
"""

if __name__ == "__main__":
    import sys
    import argparse
    desc = ""
    usage = "%(prog)s [-h] [positional] [options]"
    epilog = ""


    desc = "Parse list of args for bash"
    epilog = r"""Example:

    eval `cclargs.py \
           -D ":" \
           -desc "Script desctiption" \
           -epilog "Help Epilog message" \
           $0 \
           "-opt1" "val1"   "val2" "[desc]" \
           "-opt2" "=-val1" "val2" "[desc]" \
           ....
           ++ $*`
    """
    if sys.argv[1] in ('-h', '--help'):
        print("\n"+desc+"\n\n"+epilog)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=epilog,
        add_help=False)
    parser.add_argument("-D", dest="delimiter",
                        type=str, default=":",
                        help="Delimiter for multi arg values")
    parser.add_argument("--desc", dest="desc",
                        type=str, default="",
                        help="")
    parser.add_argument("--epilog", dest="epilog",
                        type=str, default="",
                        help="")
    (args, unknown) = parser.parse_known_args()
    sys.argv = unknown

    options = []
    delimiter = args.delimiter
    desc = args.desc
    epilog = args.epilog
    name0 = "?"
    if sys.argv[0][0] != '-':
        name0 = sys.argv.pop(0)

    while sys.argv:
        v = sys.argv.pop(0)
        if v == '++':
            break
        if len(v) == 0 or v[0] != '-':
            sys.stderr.write("cclargs, Ignoring: "+v+"\n")
            continue
        ## (d1, d2, dd) = sys.argv[0:3]
        ## del sys.argv[0:3]
        (d1, d2, dd) = ("", "", "")
        if not (len(sys.argv[0]) and sys.argv[0][0] == '-'):
            d1 = sys.argv.pop(0)
            d1 = d1[1:] if (len(d1) > 1 and d1[0:2] == '=-') else d1
        if not (len(sys.argv[0]) and sys.argv[0][0] == '-'):
            d2 = sys.argv.pop(0)
            d2 = d2[1:] if (len(d2) > 1 and d2[0:2] == '=-') else d2
        if not (len(sys.argv[0]) and sys.argv[0][0] == '-'):
            dd = sys.argv.pop(0).strip()
            if dd[0] == '[':
                dd = dd[1:]
            if dd[-1] == ']':
                dd = dd[:-1]
        options.append([v[1:].strip(), d1, d2, dd.strip()])

    parser = argparse.ArgumentParser(
        prog=name0,
        description=desc,
        usage=usage,
        epilog=epilog,
        add_help=False)

    parser.add_argument("positional",
                        nargs='*', type=str, default=None,
                        help="Positional arguments")
    parser.add_argument("-h", "--help", dest="help",
                        action="store_true",
                        help="Print this help/usage message")

    for (nn, v1, v2, dd) in options:
        dest=nn
        if nn[0] == "_":
            nn = nn[1:]
        if len(nn) == 1:
            parser.add_argument("-"+nn, dest=dest,
                                nargs='*', type=str, default=None,
                                metavar='',
                                help=dd+' ['+v1+':'+v2+']')
        else:
            parser.add_argument("--"+nn, dest=dest,
                                nargs='*', type=str, default=None,
                                metavar='',
                                help=dd+' ['+v1+':'+v2+']')

    ## convert from - to --
    sys.argv = ['-'+a if (len(a) > 2 and a[0] == '-' and a[1] != '-') else a for a in sys.argv]
    sys.argv = [name0] + sys.argv

    ## Parse provided args
    (args, unknown) = parser.parse_known_args()

    # Special case for -h
    if args.help:
        parser.print_help(sys.stderr)
        sys.stdout.write("exit 1;")
        sys.exit(1)

    # Special cases for unknown args
    if unknown and len(unknown):
        sys.stderr.write("ERROR: ("+name0+") unrecognized arguments: "+" ".join(unknown)+"\n")
        sys.stdout.write("exit 1;")
        sys.exit(1)

    # Print option=value to be evaluated by the shell
    optnamelist = [opt[0] for opt in options]
    optnamelistout = [name if name[0] == '_' else '' for name in optnamelist]
    sys.stdout.write("CCLARGS_OUT_KEYS='"+
                     " ".join(optnamelistout)
                     +"'; ")
    sys.stdout.write("CCLARGS_KEYS='"+
                     " ".join(optnamelist)
                     +"'; ")
    options = [['positional', '', '', '']] + options
    for (nn, v1, v2, dd) in options:
        v = getattr(args,nn)
        if v is None:
            v = v1
        elif len(v) == 0:
            v = v2
        else:
            v = [a[1:] if (len(a) > 1 and a[0:2] == '=-') else a for a in v]
            v = " ".join(v).replace(delimiter," ")
        if nn == 'positional':
            #TODO: check if "set -- $*" should be there instead of "set ..."
            sys.stdout.write("set nil ; shift ;  set -- $* "+v+" ; ")
        else:
            sys.stdout.write(nn+"='"+v+"'; ")
