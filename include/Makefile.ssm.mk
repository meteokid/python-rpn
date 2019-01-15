ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/include/Makefile.ssm.mk)
$(info ## )
endif

#----  SSM build Support ------------------------------------------------

GEM_SSMALL_NAME  = gem$(GEM_SFX)_$(GEM_VERSION)_all
GEM_SSMARCH_NAME = gem$(GEM_SFX)+$(COMP_ARCH)_$(GEM_VERSION)_$(SSMARCH)
GEM_SSMALL_FILES  = $(GEM_SSMALL_NAME).ssm
GEM_SSMARCH_FILES = $(GEM_SSMARCH_NAME).ssm

GEM_SSM_RELDIRBNDL = GEM/$(GEM_VERSION_X)

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
GEM_SSM_BASE_DOM  = $(SSM_BASE)/GEM/d/$(GEM_VERSION_X)gem
GEM_SSM_BASE_BNDL = $(SSM_BASE)/GEM/$(GEM_VERSION_X)gem
GEM_INSTALL   = gem_install
GEM_UNINSTALL = gem_uninstall

.PHONY: gem_ssm gem_ssm_all.ssm rm_gem_ssm_all.ssm gem_ssm_all rm_gem_ssm_all gem_ssm_arch.ssm rm_gem_ssm_arch.ssm gem_ssm_arch gem_ssm_arch_rm
gem_ssm: gem_ssm_all.ssm gem_ssm_arch.ssm
rm_gem_ssm: rm_gem_ssm_all.ssm rm_gem_ssm_all rm_gem_ssm_arch.ssm gem_ssm_arch_rm

gem_ssm_all.ssm: $(GEM_SSMALL_FILES)
$(GEM_SSMALL_FILES): gem_ssm_all rm_gem_ssm_all.ssm $(SSM_DEPOT_DIR)/$(GEM_SSMALL_NAME).ssm
rm_gem_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(GEM_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(GEM_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gem_ssm_all: rm_gem_ssm_all $(BUILDSSM)/$(GEM_SSMALL_NAME)
rm_gem_ssm_all:
	rm -rf $(BUILDSSM)/$(GEM_SSMALL_NAME)
$(BUILDSSM)/$(GEM_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_gem)/.ssm.d/exclude $(DIRORIG_gem)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control gem $(GEM_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control
	.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) sh
	export ATM_MODEL_BNDL="$(GEM_SSM_RELDIRBNDL)$(GEM_VERSION)" ;\
	export ATM_MODEL_DSTP="`date '+%Y-%m-%d %H:%M %Z'`" ;\
	cat $@/bin/.env_setup.dot0 \
	| sed "s|__MY_BNDL__|$${ATM_MODEL_BNDL#./}|" \
	| sed "s|__MY_DSTP__|$${ATM_MODEL_DSTP}|" \
	> $@/bin/.env_setup.dot

gem_ssm_arch.ssm: $(GEM_SSMARCH_FILES)
$(GEM_SSMARCH_FILES): gem_ssm_arch rm_gem_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(GEM_SSMARCH_NAME).ssm
rm_gem_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(GEM_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(GEM_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gem_ssm_arch: gem_ssm_arch_rm $(BUILDSSM)/$(GEM_SSMARCH_NAME)
gem_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(GEM_SSMARCH_NAME)
$(BUILDSSM)/$(GEM_SSMARCH_NAME):
	mkdir -p $@/include/$(EC_ARCH) $@/lib/$(EC_ARCH) $@/bin/$(BASE_ARCH) ; \
	ln -s $(EC_ARCH) $@/include/$(COMP_ARCH) ; \
	ln -s $(EC_ARCH) $@/lib/$(COMP_ARCH) ; \
	touch $@/include/dummy_$(GEM_SSMARCH_NAME).inc ; \
	touch $@/lib/libdummy_$(GEM_SSMARCH_NAME).a ; \
	touch $@/bin/dummy_$(GEM_SSMARCH_NAME).bin ; \
	echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
	echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
	echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
	echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
	cd $(MODDIR) ; \
	cp $(GEM_MOD_FILES) $@/include/$(EC_ARCH) ; \
	cd $(LIBDIR) ; \
	rsync -av `ls libgem.a libgem.a.fl libgem.so libgem_*.a libgem_*.a.fl libgem_*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) sh ; \
	cd $(BINDIR) ; \
	cp $(GEM_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	mv $@/bin/$(BASE_ARCH)/$(mainntr) $@/bin/$(BASE_ARCH)/$(mainntr_rel) ;\
	mv $@/bin/$(BASE_ARCH)/$(maindm)  $@/bin/$(BASE_ARCH)/$(maindm_rel) ;\
	cp -R $(DIRORIG_gem)/.ssm.d $@/ ; \
	.rdemk_ssm_control gem $(GEM_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control 


.PHONY: gem_install gem_uninstall
gem_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(GEM_SSM_BASE_DOM)/gem_$(GEM_VERSION) \
			--bndl=$(GEM_SSM_BASE_BNDL)/$(GEM_VERSION).bndl \
			--pre=$(gem)/ssmusedep.bndl \
			--post=$(gem)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			gem{_,+*_,-d+*_}$(GEM_VERSION)_*.ssm

gem_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(GEM_SSM_BASE_DOM)/gem_$(GEM_VERSION) \
			--bndl=$(GEM_SSM_BASE_BNDL)/$(GEM_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$gem/include/Makefile.ssm.mk [END] ========================)
endif
