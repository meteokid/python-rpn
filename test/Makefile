include ../Makefile_base
include ../$(EC_ARCH)/Makefile.inc

BASEDIR = $(PWD)/..

PYTESTS = $(shell ls *.py)

default: test
all:

test:
	for i in $(PYTESTS); \
	do  echo -e  "\n==== PY UnitTest: "  $$i "====\n"; python -d -W action:message:category:module:lineno $$i ;\
	done

