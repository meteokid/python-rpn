ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$scm/include/Makefile.ssm.mk)
$(info ## )
endif

#----  SSM build Support ------------------------------------------------

SCM_SSMALL_NAME  = scm$(SCM_SFX)_$(SCM_VERSION)_all
SCM_SSMARCH_NAME = scm$(SCM_SFX)+$(COMP_ARCH)_$(SCM_VERSION)_$(SSMARCH)
SCM_SSMALL_FILES  = $(SCM_SSMALL_NAME).ssm
SCM_SSMARCH_FILES = $(SCM_SSMARCH_NAME).ssm

SCM_SSM_RELDIRBNDL = SCM/$(SCM_VERSION_X)

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
SCM_SSM_BASE_DOM  = $(SSM_BASE)/SCM/d/$(SCM_VERSION_X)scm
SCM_SSM_BASE_BNDL = $(SSM_BASE)/SCM/$(SCM_VERSION_X)
SCM_INSTALL   = scm_install
SCM_UNINSTALL = scm_uninstall

.PHONY: scm_ssm scm_ssm_all.ssm rm_scm_ssm_all.ssm scm_ssm_all rm_scm_ssm_all scm_ssm_arch.ssm rm_scm_ssm_arch.ssm scm_ssm_arch scm_ssm_arch_rm
scm_ssm: scm_ssm_all.ssm scm_ssm_arch.ssm
rm_scm_ssm: rm_scm_ssm_all.ssm rm_scm_ssm_all rm_scm_ssm_arch.ssm scm_ssm_arch_rm

scm_ssm_all.ssm: $(SCM_SSMALL_FILES)
$(SCM_SSMALL_FILES): scm_ssm_all rm_scm_ssm_all.ssm $(SSM_DEPOT_DIR)/$(SCM_SSMALL_NAME).ssm
rm_scm_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(SCM_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(SCM_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

scm_ssm_all: rm_scm_ssm_all $(BUILDSSM)/$(SCM_SSMALL_NAME)
rm_scm_ssm_all:
	rm -rf $(BUILDSSM)/$(SCM_SSMALL_NAME)
$(BUILDSSM)/$(SCM_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_scm)/.ssm.d/exclude $(DIRORIG_scm)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control scm $(SCM_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control
	.rdemkversionfile scm $(SCM_VERSION) $@/include/$(EC_ARCH) sh
	export ATM_MODEL_BNDL="$(SCM_SSM_RELDIRBNDL)$(SCM_VERSION)" ;\
	export ATM_MODEL_DSTP="`date '+%Y-%m-%d %H:%M %Z'`" ;\
	cat $@/bin/.env_setup.dot0 \
	| sed "s|__MY_BNDL__|$${ATM_MODEL_BNDL#./}|" \
	| sed "s|__MY_DSTP__|$${ATM_MODEL_DSTP}|" \
	> $@/bin/.env_setup.dot

scm_ssm_arch.ssm: $(SCM_SSMARCH_FILES)
$(SCM_SSMARCH_FILES): scm_ssm_arch rm_scm_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(SCM_SSMARCH_NAME).ssm
rm_scm_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(SCM_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(SCM_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

scm_ssm_arch: scm_ssm_arch_rm $(BUILDSSM)/$(SCM_SSMARCH_NAME)
scm_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(SCM_SSMARCH_NAME)
$(BUILDSSM)/$(SCM_SSMARCH_NAME):
	mkdir -p $@/include/$(EC_ARCH) $@/lib/$(EC_ARCH) $@/bin/$(BASE_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/include/dummy_$(SCM_SSMARCH_NAME).inc ; \
	touch $@/lib/libdummy_$(SCM_SSMARCH_NAME).a ; \
	touch $@/bin/dummy_$(SCM_SSMARCH_NAME).bin ; \
	echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
	echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
	echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
	echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
	cd $(MODDIR) ; \
	cp $(SCM_MOD_FILES) $@/include/$(EC_ARCH) ; \
	cd $(LIBDIR) ; \
	rsync -av `ls libscm.a libscm.a.fl libscm.so libscm_*.a libscm_*.a.fl libscm_*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	.rdemkversionfile scm $(SCM_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile scm $(SCM_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile scm $(SCM_VERSION) $@/include/$(EC_ARCH) sh ; \
	cd $(BINDIR) ; \
	cp $(SCM_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	mv $@/bin/$(BASE_ARCH)/$(mainscm)  $@/bin/$(BASE_ARCH)/$(mainscm_rel) ;\
	cp -R $(DIRORIG_scm)/.ssm.d $@/ ; \
	.rdemk_ssm_control scm $(SCM_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

scm_check_bndl:
	scmdyn_version_bndl="`cat $(scm)/ssmusedep.bndl | grep scmdyn`" ;\
	if [[ $${scmdyn_version_bndl##*_} != $(SCMDYN_VERSION) ]] ; then \
		echo "ERROR: SCMDYN Version Mismatch, bndl:$${scmdyn_version_bndl##*_} != loaded:$(SCMDYN_VERSION)" ;\
		exit 1 ;\
	fi
	rpnphy_version_bndl="`cat $(scm)/ssmusedep.bndl | grep rpnphy`" ;\
	if [[ $${rpnphy_version_bndl##*_} != $(RPNPHY_VERSION) ]] ; then \
		echo "ERROR: RPNPHY Version Mismatch, bndl:$${rpnphy_version_bndl##*_} != loaded:$(RPNPHY_VERSION)" ;\
		exit 1 ;\
	fi
	modelutils_version_bndl="`cat $(scm)/ssmusedep.bndl | grep modelutils`" ;\
	modelutils_version_bndl="$${modelutils_version_bndl##*/}" ;\
	if [[ $${modelutils_version_bndl##*_} != $(MODELUTILS_VERSION) ]] ; then \
		echo "ERROR: MODELUTILS Version Mismatch, bndl:$${modelutils_version_bndl##*_} != loaded:$(MODELUTILS_VERSION)" ;\
		exit 1 ;\
	fi
	echo "OK scm_check_bndl"
	#TODO: Check modelutils, rpnphy, scmdyn, scm, vgrid, compiler... consistency

.PHONY: scm_install scm_uninstall
scm_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git \
			--dest=$(SCM_SSM_BASE_DOM)/scm_$(SCM_VERSION) \
			--bndl=$(SCM_SSM_BASE_BNDL)/$(SCM_VERSION).bndl \
			--pre=$(scm)/ssmusedep.bndl \
			--post=$(scm)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			scm{_,+*_,-d+*_}$(SCM_VERSION)_*.ssm

scm_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(SCM_SSM_BASE_DOM)/scm_$(SCM_VERSION) \
			--bndl=$(SCM_SSM_BASE_BNDL)/$(SCM_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$scm/include/Makefile.ssm.mk [END] ========================)
endif
