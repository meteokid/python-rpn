ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$modelutils/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

MODELUTILS_SSMALL_NAME  = modelutils$(MODELUTILS_SFX)_$(MODELUTILS_VERSION)_all
MODELUTILS_SSMARCH_NAME = modelutils$(MODELUTILS_SFX)+$(COMP_ARCH)_$(MODELUTILS_VERSION)_$(SSMARCH)
MODELUTILS_SSMALL_FILES  = $(MODELUTILS_SSMALL_NAME).ssm
MODELUTILS_SSMARCH_FILES = $(MODELUTILS_SSMARCH_NAME).ssm

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
MODELUTILS_SSM_BASE_DOM  = $(SSM_BASE)/ENV/d/$(MODELUTILS_VERSION_X)modelutils
MODELUTILS_SSM_BASE_BNDL = $(SSM_BASE)/ENV/$(MODELUTILS_VERSION_X)modelutils
MODELUTILS_INSTALL   = modelutils_install
MODELUTILS_UNINSTALL = modelutils_uninstall

.PHONY: modelutils_ssm modelutils_ssm_all.ssm rm_modelutils_ssm_all.ssm modelutils_ssm_all rm_modelutils_ssm_all modelutils_ssm_arch.ssm rm_modelutils_ssm_arch.ssm modelutils_ssm_arch modelutils_ssm_arch_rm
modelutils_ssm: modelutils_ssm_all.ssm modelutils_ssm_arch.ssm
rm_modelutils_ssm: rm_modelutils_ssm_all.ssm rm_modelutils_ssm_all rm_modelutils_ssm_arch.ssm modelutils_ssm_arch_rm

modelutils_ssm_all.ssm: $(MODELUTILS_SSMALL_FILES)
$(MODELUTILS_SSMALL_FILES): modelutils_ssm_all rm_modelutils_ssm_all.ssm $(SSM_DEPOT_DIR)/$(MODELUTILS_SSMALL_NAME).ssm
rm_modelutils_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(MODELUTILS_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(MODELUTILS_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

modelutils_ssm_all: rm_modelutils_ssm_all $(BUILDSSM)/$(MODELUTILS_SSMALL_NAME)
rm_modelutils_ssm_all:
	rm -rf $(BUILDSSM)/$(MODELUTILS_SSMALL_NAME)
$(BUILDSSM)/$(MODELUTILS_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_modelutils)/.ssm.d/exclude $(DIRORIG_modelutils)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control modelutils $(MODELUTILS_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

modelutils_ssm_arch.ssm: $(MODELUTILS_SSMARCH_FILES)
$(MODELUTILS_SSMARCH_FILES): modelutils_ssm_arch rm_modelutils_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(MODELUTILS_SSMARCH_NAME).ssm
rm_modelutils_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(MODELUTILS_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(MODELUTILS_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

modelutils_ssm_arch: modelutils_ssm_arch_rm $(BUILDSSM)/$(MODELUTILS_SSMARCH_NAME)
modelutils_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(MODELUTILS_SSMARCH_NAME)
$(BUILDSSM)/$(MODELUTILS_SSMARCH_NAME):
	mkdir -p $@/lib/$(EC_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/lib/libdummy_$(MODELUTILS_SSMARCH_NAME).a ; \
	cd $(LIBDIR) ; \
	rsync -av `ls libmodelutils*.a libmodelutils*.a.fl libmodelutils*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	if [[ x$(MAKE_SSM_NOMOD) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
		touch $@/include/dummy_$(MODELUTILS_SSMARCH_NAME).inc ; \
		cd $(MODDIR) ; \
		cp $(MODELUTILS_MOD_FILES) $@/include/$(EC_ARCH) ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOINC) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		touch $@/include/dummy_$(MODELUTILS_SSMARCH_NAME).mod ; \
		echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
		.rdemkversionfile modelutils $(MODELUTILS_VERSION) $@/include/$(EC_ARCH) f ; \
		.rdemkversionfile modelutils $(MODELUTILS_VERSION) $@/include/$(EC_ARCH) c ; \
		.rdemkversionfile modelutils $(MODELUTILS_VERSION) $@/include/$(EC_ARCH) sh ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOABS) != x1 ]] ; then \
		mkdir -p $@/bin/$(BASE_ARCH) ; \
		touch $@/bin/dummy_$(MODELUTILS_SSMARCH_NAME).bin ; \
		cd $(BINDIR) ; \
		cp $(MODELUTILS_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	fi ; \
	cp -R $(DIRORIG_modelutils)/.ssm.d $@/ ; \
	.rdemk_ssm_control modelutils $(MODELUTILS_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control 


.PHONY: modelutils_install modelutils_uninstall
#TODO: install all pkg should be a git repos
modelutils_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git \
			--dest=$(MODELUTILS_SSM_BASE_DOM)/modelutils_$(MODELUTILS_VERSION) \
			--bndl=$(MODELUTILS_SSM_BASE_BNDL)/$(MODELUTILS_VERSION).bndl \
			--pre=$(modelutils)/ssmusedep.bndl \
			--post=$(modelutils)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			modelutils{_,+*_,-d+*_}$(MODELUTILS_VERSION)_*.ssm

modelutils_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(MODELUTILS_SSM_BASE_DOM)/modelutils_$(MODELUTILS_VERSION) \
			--bndl=$(MODELUTILS_SSM_BASE_BNDL)/$(MODELUTILS_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$modelutils/include/Makefile.ssm.mk [END] ====================)
endif
