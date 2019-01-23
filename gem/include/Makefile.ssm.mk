ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

GEM_SSMALL_NAME  = gem$(GEM_SFX)_$(GEM_VERSION)_all
GEM_SSMARCH_NAME = gem$(GEM_SFX)+$(COMP_ARCH)_$(GEM_VERSION)_$(SSMARCH)
GEM_SSMALL_FILES  = $(GEM_SSMALL_NAME).ssm
GEM_SSMARCH_FILES = $(GEM_SSMARCH_NAME).ssm

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
   GEM_VERSION_X = test/
   SSM_TEST_INSTALL_RELDIR = test/
endif

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
GEM_SSM_BASE_DOM  = $(SSM_BASE)/GEM/d/$(GEM_VERSION_X)gem
GEM_SSM_BASE_BNDL = $(SSM_BASE)/GEM/$(GEM_VERSION_X)gem
GEM_INSTALL   = gem_install
GEM_UNINSTALL = gem_uninstall

GEM_SSM_RELDIRBNDL = GEM/$(GEM_VERSION_X)

ifeq (,$(DIRORIG_gem))
DIRORIG_gem := $(gem)
endif

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
$(BUILDSSM)/$(GEM_SSMALL_NAME): gem_ssmusedep_bndl atm_model_bndl atm_model_dstp
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_gem)/.ssm.d/exclude $(DIRORIG_gem)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control gem $(GEM_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control
	.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) sh

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
	mkdir -p $@/lib/$(EC_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/lib/libdummy_$(GEM_SSMARCH_NAME).a ; \
	cd $(LIBDIR) ; \
	rsync -av `ls libgem*.a libgem*.a.fl libgem*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	if [[ x$(MAKE_SSM_NOMOD) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
		touch $@/include/dummy_$(GEM_SSMARCH_NAME).inc ; \
		cd $(MODDIR) ; \
		cp $(GEM_MOD_FILES) $@/include/$(EC_ARCH) 2>/dev/null || true; \
	fi ; \
	if [[ x$(MAKE_SSM_NOINC) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		touch $@/include/dummy_$(GEM_SSMARCH_NAME).mod ; \
		echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
		.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) f ; \
		.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) c ; \
		.rdemkversionfile gem $(GEM_VERSION) $@/include/$(EC_ARCH) sh ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOABS) != x1 ]] ; then \
		mkdir -p $@/bin/$(BASE_ARCH) ; \
		touch $@/bin/dummy_$(GEM_SSMARCH_NAME).bin ; \
		cd $(BINDIR) ; \
		cp $(GEM_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
		mv $@/bin/$(BASE_ARCH)/$(maindm)  $@/bin/$(BASE_ARCH)/$(maindm_rel) ;\
	fi ; \
	cp -R $(DIRORIG_gem)/.ssm.d $@/ ; \
	.rdemk_ssm_control gem $(GEM_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control


.PHONY: gem_ssmusedep_bndl gem_ssmusedep_bndl_rm gem_ssmusedep_bndl_all
gem_ssmusedep_bndl: | gem_ssmusedep_bndl_rm gem_ssmusedep_bndl_all
gem_ssmusedep_bndl_rm:
	rm -f $(gem)/ssmusedep.bndl $(gem)/ssmusedep_post.bndl
gem_ssmusedep_bndl_all: $(gem)/ssmusedep.bndl $(gem)/ssmusedep_post.bndl
	ls -l $(gem)/ssmusedep.bndl $(gem)/ssmusedep_post.bndl
$(gem)/ssmusedep.bndl:
	touch $@ ;\
	if [[ -f $(gem)/DEPENDENCIES.external.bndl ]] ; then \
	   cat $(gem)/DEPENDENCIES.external.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(gem)/DEPENDENCIES.external.$${RDENETWORK}.bndl ]] ; then \
	   cat $(gem)/DEPENDENCIES.external.$${RDENETWORK}.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(gem)/DEPENDENCIES.mig.bndl ]] ; then \
	   for i in `cat $(gem)/DEPENDENCIES.mig.bndl | tr '\n' ' '` ; do \
	      i2="$$(echo $$i | sed 's|/x/|/|g')" ;\
	      if [[ "x$(SSM_TEST_INSTALL_RELDIR)" == "x" ]] ; then i2=$$i ; fi ;\
	      i0=$${i2%/*} ;\
	      i1=`echo $${i2} | sed "s|$${i0%/*}/||"` ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} >> $@ ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} ;\
	   done ;\
	fi
$(gem)/ssmusedep_post.bndl:
	touch $@
	if [[ -f $(gem)/DEPENDENCIES.post.bndl ]] ; then \
	   cat $(gem)/DEPENDENCIES.post.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(gem)/DEPENDENCIES.post.$${RDENETWORK}.bndl ]] ; then \
	   cat $(gem)/DEPENDENCIES.post.$${RDENETWORK}.bndl >> $@ ;\
	fi ;\

atm_model_bndl: | atm_model_bndl_rm $(gem)/ATM_MODEL_BNDL
atm_model_bndl_rm:
	rm -f $(gem)/ATM_MODEL_BNDL
$(gem)/ATM_MODEL_BNDL:
	echo "$(GEM_SSM_RELDIRBNDL)$(GEM_VERSION)" > $@

atm_model_dstp: | atm_model_dstp_rm $(gem)/ATM_MODEL_DSTP
atm_model_dstp_rm:
	rm -f $(gem)/ATM_MODEL_DSTP
$(gem)/ATM_MODEL_DSTP:
	echo "`date '+%Y-%m-%d %H:%M %Z'`" > $@

# gem_check_bndl:
# 	gemdyn_version_bndl="`cat $(gem)/ssmusedep.bndl | grep gemdyn`" ;\
# 	if [[ $${gemdyn_version_bndl##*_} != $(GEMDYN_VERSION) ]] ; then \
# 		echo "ERROR: GEMDYN Version Mismatch, bndl:$${gemdyn_version_bndl##*_} != loaded:$(GEMDYN_VERSION)" ;\
# 		exit 1 ;\
# 	fi
# 	rpnphy_version_bndl="`cat $(gem)/ssmusedep.bndl | grep rpnphy`" ;\
# 	if [[ $${rpnphy_version_bndl##*_} != $(RPNPHY_VERSION) ]] ; then \
# 		echo "ERROR: RPNPHY Version Mismatch, bndl:$${rpnphy_version_bndl##*_} != loaded:$(RPNPHY_VERSION)" ;\
# 		exit 1 ;\
# 	fi
# 	modelutils_version_bndl="`cat $(gem)/ssmusedep.bndl | grep modelutils`" ;\
# 	modelutils_version_bndl="$${modelutils_version_bndl##*/}" ;\
# 	if [[ $${modelutils_version_bndl##*_} != $(MODELUTILS_VERSION) ]] ; then \
# 		echo "ERROR: MODELUTILS Version Mismatch, bndl:$${modelutils_version_bndl##*_} != loaded:$(MODELUTILS_VERSION)" ;\
# 		exit 1 ;\
# 	fi
# 	echo "OK gem_check_bndl"
# 	#TODO: Check modelutils, rpnphy, gemdyn, gem, vgrid, compiler... consistency


.PHONY: gem_install gem_uninstall
gem_install: gem_ssmusedep_bndl
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git $(SSM_SKIP_INSTALLED) \
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
$(info ## ==== $$gem/include/Makefile.ssm.mk [END] ====================)
endif
