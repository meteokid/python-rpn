include ../Makefile_base
include ../Makefile_$(ARCH)

BASEDIR = $(PWD)/..

PYTESTS = $(shell ls *.py)

default: test
all:

test:
	for i in $(PYTESTS); \
	do  echo -e  "\n==== PY UnitTest: "  $$i "====\n"; python $$i ;\
	done

