ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$modelscripts/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

MODELSCRIPTS_SSMALL_NAME  = modelscripts$(MODELSCRIPTS_SFX)_$(MODELSCRIPTS_VERSION)_all
MODELSCRIPTS_SSMARCH_NAME = modelscripts$(MODELSCRIPTS_SFX)+$(COMP_ARCH)_$(MODELSCRIPTS_VERSION)_$(SSMARCH)
MODELSCRIPTS_SSMALL_FILES  = $(MODELSCRIPTS_SSMALL_NAME).ssm
MODELSCRIPTS_SSMARCH_FILES =  ## $(MODELSCRIPTS_SSMARCH_NAME).ssm

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
   MODELSCRIPTS_VERSION_X = test/
   SSM_TEST_INSTALL_RELDIR = test/
endif

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
MODELSCRIPTS_SSM_BASE_DOM  = $(SSM_BASE)/GEM/d/$(MODELSCRIPTS_VERSION_X)modelscripts
MODELSCRIPTS_SSM_BASE_BNDL = $(SSM_BASE)/GEM/$(MODELSCRIPTS_VERSION_X)modelscripts
MODELSCRIPTS_INSTALL   = modelscripts_install
MODELSCRIPTS_UNINSTALL = modelscripts_uninstall

DIRORIG_modelscripts := $(modelscripts)
# BUILDSSM = $(TMPDIR)/build-modelscripts
BUILDSSM = $(ROOT)/$(CONST_BUILDSSM)

.PHONY: 
modelscripts_ssm: modelscripts_ssm_all.ssm modelscripts_ssm_arch.ssm
rm_modelscripts_ssm: rm_modelscripts_ssm_all.ssm rm_modelscripts_ssm_all rm_modelscripts_ssm_arch.ssm modelscripts_ssm_arch_rm

modelscripts_ssm_all.ssm: $(MODELSCRIPTS_SSMALL_FILES)
$(MODELSCRIPTS_SSMALL_FILES): modelscripts_ssm_all rm_modelscripts_ssm_all.ssm $(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMALL_NAME).ssm
rm_modelscripts_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

modelscripts_ssm_all: rm_modelscripts_ssm_all $(BUILDSSM)/$(MODELSCRIPTS_SSMALL_NAME)
rm_modelscripts_ssm_all:
	rm -rf $(BUILDSSM)/$(MODELSCRIPTS_SSMALL_NAME)
$(BUILDSSM)/$(MODELSCRIPTS_SSMALL_NAME): modelscripts_ssmusedep_bndl
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_modelscripts)/.ssm.d/exclude $(DIRORIG_modelscripts)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	echo "CURRENT_VERSION=$(MODELSCRIPTS_VERSION)" > $@/share/gem-mod/mod/VERSION ; \
	.rdemk_ssm_control modelscripts $(MODELSCRIPTS_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

modelscripts_ssm_arch.ssm: $(MODELSCRIPTS_SSMARCH_FILES)
$(MODELSCRIPTS_SSMARCH_FILES): modelscripts_ssm_arch rm_modelscripts_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMARCH_NAME).ssm
rm_modelscripts_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

modelscripts_ssm_arch: modelscripts_ssm_arch_rm $(BUILDSSM)/$(MODELSCRIPTS_SSMARCH_NAME)
modelscripts_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(MODELSCRIPTS_SSMARCH_NAME)
$(BUILDSSM)/$(MODELSCRIPTS_SSMARCH_NAME):
	mkdir -p $@/lib/$(EC_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/lib/libdummy_$(MODELSCRIPTS_SSMARCH_NAME).a ; \
	cd $(LIBDIR) ; \
	rsync -av `ls libmodelscripts*.a libmodelscripts*.a.fl libmodelscripts*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	if [[ x$(MAKE_SSM_NOMOD) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
		touch $@/include/dummy_$(MODELSCRIPTS_SSMARCH_NAME).inc ; \
		cd $(MODDIR) ; \
		cp $(MODELSCRIPTS_MOD_FILES) $@/include/$(EC_ARCH) ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOINC) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		touch $@/include/dummy_$(MODELSCRIPTS_SSMARCH_NAME).mod ; \
		echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
		.rdemkversionfile modelscripts $(MODELSCRIPTS_VERSION) $@/include/$(EC_ARCH) f ; \
		.rdemkversionfile modelscripts $(MODELSCRIPTS_VERSION) $@/include/$(EC_ARCH) c ; \
		.rdemkversionfile modelscripts $(MODELSCRIPTS_VERSION) $@/include/$(EC_ARCH) sh ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOABS) != x1 ]] ; then \
		mkdir -p $@/bin/$(BASE_ARCH) ; \
		touch $@/bin/dummy_$(MODELSCRIPTS_SSMARCH_NAME).bin ; \
		cd $(BINDIR) ; \
		cp $(MODELSCRIPTS_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	fi ; \
	cp -R $(DIRORIG_modelscripts)/.ssm.d $@/ ; \
	.rdemk_ssm_control modelscripts $(MODELSCRIPTS_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control


.PHONY: modelscripts_ssmusedep_bndl modelscripts_ssmusedep_bndl_rm modelscripts_ssmusedep_bndl_all
modelscripts_ssmusedep_bndl: | modelscripts_ssmusedep_bndl_rm modelscripts_ssmusedep_bndl_all
modelscripts_ssmusedep_bndl_rm:
	rm -f $(modelscripts)/ssmusedep.bndl $(modelscripts)/ssmusedep_post.bndl
modelscripts_ssmusedep_bndl_all: $(modelscripts)/ssmusedep.bndl $(modelscripts)/ssmusedep_post.bndl
	ls -l $(modelscripts)/ssmusedep.bndl $(modelscripts)/ssmusedep_post.bndl
$(modelscripts)/ssmusedep.bndl:
	touch $@ ;\
	if [[ -f $(modelscripts)/DEPENDENCIES.external.bndl ]] ; then \
	   cat $(modelscripts)/DEPENDENCIES.external.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(modelscripts)/DEPENDENCIES.external.$${RDENETWORK}.bndl ]] ; then \
	   cat $(modelscripts)/DEPENDENCIES.external.$${RDENETWORK}.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(modelscripts)/DEPENDENCIES.mig.bndl ]] ; then \
	   for i in `cat $(modelscripts)/DEPENDENCIES.mig.bndl | tr '\n' ' '` ; do \
	      i2="$$(echo $$i | sed 's|/x/|/|g')" ;\
	      if [[ "x$(SSM_TEST_INSTALL_RELDIR)" == "x" ]] ; then i2=$$i ; fi ;\
	      i0=$${i2%/*} ;\
	      i1=`echo $${i2} | sed "s|$${i0%/*}/||"` ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} >> $@ ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} ;\
	   done ;\
	fi
$(modelscripts)/ssmusedep_post.bndl:
	touch $@

.PHONY: modelscripts_install modelscripts_uninstall
modelscripts_install: modelscripts_ssmusedep_bndl
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git $(SSM_SKIP_INSTALLED) \
			--dest=$(MODELSCRIPTS_SSM_BASE_DOM)/modelscripts_$(MODELSCRIPTS_VERSION) \
			--bndl=$(MODELSCRIPTS_SSM_BASE_BNDL)/$(MODELSCRIPTS_VERSION).bndl \
			--pre=$(modelscripts)/ssmusedep.bndl \
			--post=$(modelscripts)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			modelscripts{_,+*_,-d+*_}$(MODELSCRIPTS_VERSION)_*.ssm

modelscripts_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(MODELSCRIPTS_SSM_BASE_DOM)/modelscripts_$(MODELSCRIPTS_VERSION) \
			--bndl=$(MODELSCRIPTS_SSM_BASE_BNDL)/$(MODELSCRIPTS_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$modelscripts/include/Makefile.ssm.mk [END] ====================)
endif
