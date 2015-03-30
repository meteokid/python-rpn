ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnpy/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

RPNPY_SSMALL_NAME  = rpnpy_$(RPNPY_VERSION)_all
# RPNPY_SSMARCH_NAME = rpnpy+$(COMP_ARCH)_$(RPNPY_VERSION)_$(SSMARCH)
RPNPY_SSMALL_FILES  = $(RPNPY_SSMALL_NAME).ssm
# RPNPY_SSMARCH_FILES = $(RPNPY_SSMARCH_NAME).ssm

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
RPNPY_SSM_BASE_DOM  = $(SSM_BASE)/ENV/d/$(RPNPY_VERSION_X)rpnpy
RPNPY_SSM_BASE_BNDL = $(SSM_BASE)/ENV/$(RPNPY_VERSION_X)rpnpy
RPNPY_INSTALL   = rpnpy_install
RPNPY_UNINSTALL = rpnpy_uninstall

.PHONY: rpnpy_ssm rpnpy_ssm_all.ssm rm_rpnpy_ssm_all.ssm rpnpy_ssm_all rm_rpnpy_ssm_all rpnpy_ssm_arch.ssm rm_rpnpy_ssm_arch.ssm rpnpy_ssm_arch rpnpy_ssm_arch_rm
rpnpy_ssm: rpnpy_ssm_all.ssm rpnpy_ssm_arch.ssm
rm_rpnpy_ssm: rm_rpnpy_ssm_all.ssm rm_rpnpy_ssm_all rm_rpnpy_ssm_arch.ssm rpnpy_ssm_arch_rm

rpnpy_ssm_all.ssm: $(RPNPY_SSMALL_FILES)
$(RPNPY_SSMALL_FILES): rpnpy_ssm_all rm_rpnpy_ssm_all.ssm $(SSM_DEPOT_DIR)/$(RPNPY_SSMALL_NAME).ssm
rm_rpnpy_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(RPNPY_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(RPNPY_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

rpnpy_ssm_all: rm_rpnpy_ssm_all $(BUILDSSM)/$(RPNPY_SSMALL_NAME)
rm_rpnpy_ssm_all:
	rm -rf $(BUILDSSM)/$(RPNPY_SSMALL_NAME)
$(BUILDSSM)/$(RPNPY_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_rpnpy)/.ssm.d/exclude $(DIRORIG_rpnpy)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control rpnpy $(RPNPY_VERSION) "all ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

rpnpy_ssm_arch.ssm: $(RPNPY_SSMARCH_FILES)
$(RPNPY_SSMARCH_FILES): rpnpy_ssm_arch rm_rpnpy_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(RPNPY_SSMARCH_NAME).ssm
rm_rpnpy_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(RPNPY_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(RPNPY_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

rpnpy_ssm_arch: rpnpy_ssm_arch_rm $(BUILDSSM)/$(RPNPY_SSMARCH_NAME)
rpnpy_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(RPNPY_SSMARCH_NAME)
$(BUILDSSM)/$(RPNPY_SSMARCH_NAME):
	mkdir -p $@/include/$(EC_ARCH) $@/lib/$(EC_ARCH) $@/bin/$(BASE_ARCH) ; \
	echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
	echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
	echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
	echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
	cd $(MODDIR) ; \
	cp $(RPNPY_MOD_FILES) $@/include/$(EC_ARCH) ; \
	cd $(LIBDIR) ; \
	rsync -av `ls librpnpy*.a librpnpy*.a.fl librpnpy*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	.rdemkversionfile rpnpy $(RPNPY_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile rpnpy $(RPNPY_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile rpnpy $(RPNPY_VERSION) $@/include/$(EC_ARCH) sh ; \
	cd $(BINDIR) ; \
	cp $(RPNPY_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	cp -R $(DIRORIG_rpnpy)/.ssm.d $@/ ; \
	.rdemk_ssm_control rpnpy $(RPNPY_VERSION) "$(SSMORDARCH) ; $(SSMARCH) ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control 


.PHONY: rpnpy_install rpnpy_uninstall
rpnpy_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(RPNPY_SSM_BASE_DOM)/rpnpy_$(RPNPY_VERSION) \
			--bndl=$(RPNPY_SSM_BASE_BNDL)/$(RPNPY_VERSION).bndl \
			--pre=$(rpnpy)/ssmusedep.bndl \
			--post=$(rpnpy)/ssmusedep_post.bndl \
			--base=$(SSM_BASE) \
			rpnpy{_,+*_}$(RPNPY_VERSION)_*.ssm

rpnpy_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(RPNPY_SSM_BASE_DOM)/rpnpy_$(RPNPY_VERSION) \
			--bndl=$(RPNPY_SSM_BASE_BNDL)/$(RPNPY_VERSION).bndl \
			--base=$(SSM_BASE) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnpy/include/Makefile.ssm.mk [END] ====================)
endif
