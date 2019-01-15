# ATTENTION!
# YOU NEED TO USE THE GMAKE IN /users/dor/afsi/sul/ovbin/gmake for this
# Makefile to work, other versions installed are currently too old!
#
# common section definition
include config/config.mk

ssm: profile.sh
	echo "Creating SSM package $(SSMPACKAGE)..."
	ssmprep --basePath $(shell pwd)/$(SSMPACKAGE) -p $(SSMPACKAGE) --editorPath /bin/echo; \
	chmod a+x post-install; \
	cp post-install $(SSMPACKAGE)/.ssm.d; \
	cp profile.sh $(SSMPACKAGE)/.ssm.d; \
	(cd ..; find ref -path '*/.svn*' -prune -o -exec rsync -R {} ssm/$(SSMPACKAGE) \;) ; \
	(cd ..; find bin -path '*/.svn*' -prune -o -exec rsync -R {} ssm/$(SSMPACKAGE) \;) ; \
	(cd ..; find etc -path '*/.svn*' -prune -o -exec rsync -R {} ssm/$(SSMPACKAGE) \;) ; \
	(cd ..; find config -path '*/.svn*' -prune -o -exec rsync -R {} ssm/$(SSMPACKAGE) \;) ; \
	(cd $(SSMPACKAGE)/ref; mkdir sequencing logs listings modules) ; \
	echo "Package: ${SWNAME}" > $(SSMPACKAGE)/.ssm.d/control; \
	echo "Version: ${VERSION}" >> $(SSMPACKAGE)/.ssm.d/control; \
	echo "Platform: all" >> $(SSMPACKAGE)/.ssm.d/control; \
	echo "Maintainer: $(shell whoami)" >> $(SSMPACKAGE)/.ssm.d/control; \
	echo "BuildInfo:" >> $(SSMPACKAGE)/.ssm.d/control; \
	echo "Description: ${SWDESCRIP}" >> $(SSMPACKAGE)/.ssm.d/control; \
	tar cvf - $(SSMPACKAGE) | gzip -> $(SSMPACKAGE).ssm

profile.sh: profile
	echo "__version=${VERSION}" >$@ ;\
	echo "__swname=${SWNAME}" >>$@ ;\
	cat profile >>$@ ;\
	echo "export MAESTRO_CURRENT_PREFIX=${MAESTRO_CURRENT_PREFIX}" >>$@

clean:
	rm -rf $(SSMPACKAGE).ssm $(SSMPACKAGE) profile.sh

distclean: clean
	find .. -name "*~" -exec rm -f {} \;
