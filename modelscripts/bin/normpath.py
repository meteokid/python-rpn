#!/usr/bin/env python

import optparse
import os.path
import sys
def main(): 

    # Command line argument parsing
    parser = optparse.OptionParser()
    parser.add_option("-p","--path",dest="path",default=None,
                      help="Returns python-normalized PATH.)",metavar="FILE")
    (options,args) = parser.parse_args()
    
    sys.stdout.write(os.path.normpath(options.path))


if __name__ == "__main__":
    main()


