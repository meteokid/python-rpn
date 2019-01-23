ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gemdyn/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

GEMDYN_SSMALL_NAME  = gemdyn$(GEMDYN_SFX)_$(GEMDYN_VERSION)_all
GEMDYN_SSMARCH_NAME = gemdyn$(GEMDYN_SFX)+$(COMP_ARCH)_$(GEMDYN_VERSION)_$(SSMARCH)
GEMDYN_SSMALL_FILES  = $(GEMDYN_SSMALL_NAME).ssm
GEMDYN_SSMARCH_FILES = $(GEMDYN_SSMARCH_NAME).ssm

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
   GEMDYN_VERSION_X = test/
   SSM_TEST_INSTALL_RELDIR = test/
endif

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
GEMDYN_SSM_BASE_DOM  = $(SSM_BASE)/GEM/d/$(GEMDYN_VERSION_X)gemdyn
GEMDYN_SSM_BASE_BNDL = $(SSM_BASE)/GEM/$(GEMDYN_VERSION_X)gemdyn
GEMDYN_INSTALL   = gemdyn_install
GEMDYN_UNINSTALL = gemdyn_uninstall

.PHONY: gemdyn_ssm gemdyn_ssm_all.ssm rm_gemdyn_ssm_all.ssm gemdyn_ssm_all rm_gemdyn_ssm_all gemdyn_ssm_arch.ssm rm_gemdyn_ssm_arch.ssm gemdyn_ssm_arch gemdyn_ssm_arch_rm
gemdyn_ssm: gemdyn_ssm_all.ssm gemdyn_ssm_arch.ssm
rm_gemdyn_ssm: rm_gemdyn_ssm_all.ssm rm_gemdyn_ssm_all rm_gemdyn_ssm_arch.ssm gemdyn_ssm_arch_rm

gemdyn_ssm_all.ssm: $(GEMDYN_SSMALL_FILES)
$(GEMDYN_SSMALL_FILES): gemdyn_ssm_all rm_gemdyn_ssm_all.ssm $(SSM_DEPOT_DIR)/$(GEMDYN_SSMALL_NAME).ssm
rm_gemdyn_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(GEMDYN_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(GEMDYN_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gemdyn_ssm_all: rm_gemdyn_ssm_all $(BUILDSSM)/$(GEMDYN_SSMALL_NAME)
rm_gemdyn_ssm_all:
	rm -rf $(BUILDSSM)/$(GEMDYN_SSMALL_NAME)
$(BUILDSSM)/$(GEMDYN_SSMALL_NAME): gemdyn_ssmusedep_bndl
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_gemdyn)/.ssm.d/exclude $(DIRORIG_gemdyn)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

gemdyn_ssm_arch.ssm: $(GEMDYN_SSMARCH_FILES)
$(GEMDYN_SSMARCH_FILES): gemdyn_ssm_arch rm_gemdyn_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(GEMDYN_SSMARCH_NAME).ssm
rm_gemdyn_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(GEMDYN_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(GEMDYN_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gemdyn_ssm_arch: gemdyn_ssm_arch_rm $(BUILDSSM)/$(GEMDYN_SSMARCH_NAME)
gemdyn_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(GEMDYN_SSMARCH_NAME)
$(BUILDSSM)/$(GEMDYN_SSMARCH_NAME):
	mkdir -p $@/lib/$(EC_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/lib/libdummy_$(GEMDYN_SSMARCH_NAME).a ; \
	cd $(LIBDIR) ; \
	rsync -av `ls libgemdyn*.a libgemdyn*.a.fl libgemdyn*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	if [[ x$(MAKE_SSM_NOMOD) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
		touch $@/include/dummy_$(GEMDYN_SSMARCH_NAME).inc ; \
		cd $(MODDIR) ; \
		cp $(GEMDYN_MOD_FILES) $@/include/$(EC_ARCH) ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOINC) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		touch $@/include/dummy_$(GEMDYN_SSMARCH_NAME).mod ; \
		echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
		.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) f ; \
		.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) c ; \
		.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) sh ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOABS) != x1 ]] ; then \
		mkdir -p $@/bin/$(BASE_ARCH) ; \
		touch $@/bin/dummy_$(GEMDYN_SSMARCH_NAME).bin ; \
		cd $(BINDIR) ; \
		cp $(GEMDYN_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	fi ; \
	cp -R $(DIRORIG_gemdyn)/.ssm.d $@/ ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control


.PHONY: gemdyn_ssmusedep_bndl gemdyn_ssmusedep_bndl_rm gemdyn_ssmusedep_bndl_all
gemdyn_ssmusedep_bndl: | gemdyn_ssmusedep_bndl_rm gemdyn_ssmusedep_bndl_all
gemdyn_ssmusedep_bndl_rm:
	rm -f $(gemdyn)/ssmusedep.bndl $(gemdyn)/ssmusedep_post.bndl
gemdyn_ssmusedep_bndl_all: $(gemdyn)/ssmusedep.bndl $(gemdyn)/ssmusedep_post.bndl
	ls -l $(gemdyn)/ssmusedep.bndl $(gemdyn)/ssmusedep_post.bndl
$(gemdyn)/ssmusedep.bndl:
	touch $@ ;\
	if [[ -f $(gemdyn)/DEPENDENCIES.external.bndl ]] ; then \
	   cat $(gemdyn)/DEPENDENCIES.external.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(gemdyn)/DEPENDENCIES.external.$${RDENETWORK}.bndl ]] ; then \
	   cat $(gemdyn)/DEPENDENCIES.external.$${RDENETWORK}.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(gemdyn)/DEPENDENCIES.mig.bndl ]] ; then \
	   for i in `cat $(gemdyn)/DEPENDENCIES.mig.bndl | tr '\n' ' '` ; do \
	      i2="$$(echo $$i | sed 's|/x/|/|g')" ;\
	      if [[ "x$(SSM_TEST_INSTALL_RELDIR)" == "x" ]] ; then i2=$$i ; fi ;\
	      i0=$${i2%/*} ;\
	      i1=`echo $${i2} | sed "s|$${i0%/*}/||"` ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} >> $@ ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} ;\
	   done ;\
	fi
$(gemdyn)/ssmusedep_post.bndl:
	touch $@

.PHONY: gemdyn_install gemdyn_uninstall
gemdyn_install: gemdyn_ssmusedep_bndl
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git $(SSM_SKIP_INSTALLED) \
			--dest=$(GEMDYN_SSM_BASE_DOM)/gemdyn_$(GEMDYN_VERSION) \
			--bndl=$(GEMDYN_SSM_BASE_BNDL)/$(GEMDYN_VERSION).bndl \
			--pre=$(gemdyn)/ssmusedep.bndl \
			--post=$(gemdyn)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			gemdyn{_,+*_,-d+*_}$(GEMDYN_VERSION)_*.ssm

gemdyn_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(GEMDYN_SSM_BASE_DOM)/gemdyn_$(GEMDYN_VERSION) \
			--bndl=$(GEMDYN_SSM_BASE_BNDL)/$(GEMDYN_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$gemdyn/include/Makefile.ssm.mk [END] ====================)
endif
