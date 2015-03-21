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

.PHONY: gemdyn_ssm gemdyn_ssm_all.ssm rm_gemdyn_ssm_all.ssm gemdyn_ssm_all rm_gemdyn_ssm_all gemdyn_ssm_arch.ssm rm_gemdyn_ssm_arch.ssm gemdyn_ssm_arch gemdyn_ssm_arch_rm
gemdyn_ssm: gemdyn_ssm_all.ssm gemdyn_ssm_arch.ssm
rm_gemdyn_ssm: rm_gemdyn_ssm_all.ssm rm_gemdyn_ssm_all rm_gemdyn_ssm_arch.ssm gemdyn_ssm_arch_rm

gemdyn_ssm_all.ssm: $(GEMDYN_SSMALL_FILES)
$(GEMDYN_SSMALL_FILES): gemdyn_ssm_all rm_gemdyn_ssm_all.ssm $(SSMDEPOTDIR)/$(GEMDYN_SSMALL_NAME).ssm
rm_gemdyn_ssm_all.ssm:
	rm -f $(SSMDEPOTDIR)/$(GEMDYN_SSMALL_NAME).ssm
$(SSMDEPOTDIR)/$(GEMDYN_SSMALL_NAME).ssm:
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
$(GEMDYN_SSMARCH_FILES): gemdyn_ssm_arch rm_gemdyn_ssm_arch.ssm $(SSMDEPOTDIR)/$(GEMDYN_SSMARCH_NAME).ssm
rm_gemdyn_ssm_arch.ssm:
	rm -f $(SSMDEPOTDIR)/$(GEMDYN_SSMARCH_NAME).ssm
$(SSMDEPOTDIR)/$(GEMDYN_SSMARCH_NAME).ssm:
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
	.rdemkversionfile gem $(GEMDYN_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile gem $(GEMDYN_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile gem $(GEMDYN_VERSION) $@/include/$(EC_ARCH) sh ; \
	cd $(BINDIR) ; \
	cp $(GEMDYN_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	mv $@/bin/$(BASE_ARCH)/$(mainntr) $@/bin/$(BASE_ARCH)/$(mainntr_rel) ;\
	mv $@/bin/$(BASE_ARCH)/$(maindm)  $@/bin/$(BASE_ARCH)/$(maindm_rel) ;\
	cp -R $(DIRORIG_gemdyn)/.ssm.d $@/ ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) "$(SSMORDARCH) ; $(SSMARCH) ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control 

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$gemdyn/include/Makefile.ssm.mk ==============================)
endif
