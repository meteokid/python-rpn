ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$migdep/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

ifeq (,$(RDENETWORK))
   ifneq (,$(wildcard /ssm/net/*))
      RDENETWORK=cmc
   else
      RDENETWORK=science
   endif
endif
ifeq (science,$(RDENETWORK))
   ifeq (,$(SSM_PREFIX))
      SSM_PREFIX = eccc/mrd/rpn/MIG/
   endif
endif

# SSM_TEST_INSTALL = 1
ifeq (1,$(SSM_TEST_INSTALL))
   MIGDEP_VERSION_X = test/
   SSM_TEST_INSTALL_RELDIR = test/
endif

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
MIGDEP_SSM_BASE_DOM  = $(SSM_BASE)/ENV/d/$(MIGDEP_VERSION_X)migdep
MIGDEP_SSM_BASE_BNDL = $(SSM_BASE)/ENV/$(MIGDEP_VERSION_X)migdep
MIGDEP_INSTALL   = migdep_install
MIGDEP_UNINSTALL = migdep_uninstall

BUILDSSM = $(ROOT)/$(CONST_BUILDSSM)

.PHONY: migdep_ssmusedep_bndl migdep_ssmusedep_bndl_rm migdep_ssmusedep_bndl_all
migdep_ssmusedep_bndl: | migdep_ssmusedep_bndl_rm migdep_ssmusedep_bndl_all
migdep_ssmusedep_bndl_rm:
	rm -f $(migdep)/ssmusedep.bndl $(migdep)/ssmusedep_post.bndl
migdep_ssmusedep_bndl_all: $(migdep)/ssmusedep.bndl $(migdep)/ssmusedep_post.bndl
	ls -l $(migdep)/ssmusedep.bndl $(migdep)/ssmusedep_post.bndl
$(migdep)/ssmusedep.bndl:
	touch $@ ;\
	if [[ -f $(migdep)/DEPENDENCIES.external.bndl ]] ; then \
	   cat $(migdep)/DEPENDENCIES.external.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(migdep)/DEPENDENCIES.external.$${RDENETWORK}.bndl ]] ; then \
	   cat $(migdep)/DEPENDENCIES.external.$${RDENETWORK}.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(migdep)/DEPENDENCIES.mig.bndl ]] ; then \
	   for i in `cat $(migdep)/DEPENDENCIES.mig.bndl | tr '\n' ' '` ; do \
	      i2="$$(echo $$i | sed 's|/x/|/|g')" ;\
	      if [[ "x$(SSM_TEST_INSTALL_RELDIR)" == "x" ]] ; then i2=$$i ; fi ;\
	      i0=$${i2%/*} ;\
	      i1=`echo $${i2} | sed "s|$${i0%/*}/||"` ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} >> $@ ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} ;\
	   done ;\
	fi
$(migdep)/ssmusedep_post.bndl:
	touch $@

.PHONY: migdep_install migdep_uninstall
migdep_install: migdep_ssmusedep_bndl
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	mkdir -p $(MIGDEP_SSM_BASE_BNDL)
	if [[ ! -f $(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl ]] ; then \
		cp $(migdep)/ssmusedep.bndl $(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl; \
	else \
		echo "Skipping: $(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl" ; \
	fi
	# cd $(SSM_DEPOT_DIR) ;\
	# rdessm-install -v \
	# 		--git $(SSM_SKIP_INSTALLED) \
	# 		--dest=$(MIGDEP_SSM_BASE_DOM)/migdep_$(MIGDEP_VERSION) \
	# 		--bndl=$(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl \
	# 		--pre=$(migdep)/ssmusedep.bndl \
	# 		--post=$(migdep)/ssmusedep_post.bndl \
	# 		--base=$(SSM_BASE2) \
	# 		migdep{_,+*_,-d+*_}$(MIGDEP_VERSION)_*.ssm

migdep_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	if [[ -f $(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl ]] ; then \
		rm -f $(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl ;\
	fi
	# cd $(SSM_DEPOT_DIR) ;\
	# rdessm-install -v \
	# 		--dest=$(MIGDEP_SSM_BASE_DOM)/migdep_$(MIGDEP_VERSION) \
	# 		--bndl=$(MIGDEP_SSM_BASE_BNDL)/$(MIGDEP_VERSION).bndl \
	# 		--base=$(SSM_BASE2) \
	# 		--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$migdep/include/Makefile.ssm.mk [END] ====================)
endif
