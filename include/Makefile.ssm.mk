ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gemdyn/include/Makefile.ssm.mk)
$(info ## )
endif

#----  SSM build Support ------------------------------------------------

GEMDYN_SSMALL_NAME  = gemdyn_$(GEMDYN_VERSION)_all
GEMDYN_SSMARCH_NAME = gemdyn+$(COMP_ARCH)_$(GEMDYN_VERSION)_$(SSMARCH)
GEMDYN_SSMALL_FILES  = $(GEMDYN_SSMALL_NAME).ssm
GEMDYN_SSMARCH_FILES = $(GEMDYN_SSMARCH_NAME).ssm

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
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
	cd $(BUILD) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gemdyn_ssm_all: rm_gemdyn_ssm_all $(BUILD)/$(GEMDYN_SSMALL_NAME)
rm_gemdyn_ssm_all:
	rm -rf $(BUILD)/$(GEMDYN_SSMALL_NAME)
$(BUILD)/$(GEMDYN_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_gemdyn)/.ssm.d/exclude $(DIRORIG_gemdyn)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) "all ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

gemdyn_ssm_arch.ssm: $(GEMDYN_SSMARCH_FILES)
$(GEMDYN_SSMARCH_FILES): gemdyn_ssm_arch rm_gemdyn_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(GEMDYN_SSMARCH_NAME).ssm
rm_gemdyn_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(GEMDYN_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(GEMDYN_SSMARCH_NAME).ssm:
	cd $(BUILD) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gemdyn_ssm_arch: gemdyn_ssm_arch_rm $(BUILD)/$(GEMDYN_SSMARCH_NAME)
gemdyn_ssm_arch_rm:
	rm -rf $(BUILD)/$(GEMDYN_SSMARCH_NAME)
$(BUILD)/$(GEMDYN_SSMARCH_NAME):
	mkdir -p $@/include/$(EC_ARCH) $@/lib/$(EC_ARCH) $@/bin/$(BASE_ARCH) ; \
	cp $(GEMDYN_MOD_FILES) $@/include/$(EC_ARCH) ; \
	rsync -av $(wildcard libgemdyn*.a libgemdyn*.a.fl libgemdyn*.so) $@/lib/$(EC_ARCH)/ ; \
	.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) sh ; \
	cd $(BINDIR) ; \
	cp $(GEMDYN_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	cp -R $(DIRORIG_gemdyn)/.ssm.d $@/ ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) "$(SSMORDARCH) ; $(SSMARCH) ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control 


.PHONY: gemdyn_install gemdyn_uninstall
gemdyn_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(GEMDYN_SSM_BASE_DOM)/gemdyn_$(GEMDYN_VERSION) \
			--bndl=$(GEMDYN_SSM_BASE_BNDL)/$(GEMDYN_VERSION).bndl \
			--pre=$(gemdyn)/ssmusedep.bndl \
			--post=$(gemdyn)/ssmusedep_post.bndl \
			--base=$(SSM_BASE) \
			gemdyn{_,+*_}$(GEMDYN_VERSION)_*.ssm

gemdyn_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(GEMDYN_SSM_BASE_DOM)/gemdyn_$(GEMDYN_VERSION) \
			--bndl=$(GEMDYN_SSM_BASE_BNDL)/$(GEMDYN_VERSION).bndl \
			--base=$(SSM_BASE) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$gemdyn/include/Makefile.ssm.mk [END] ========================)
endif
