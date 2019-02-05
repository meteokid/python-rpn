ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnpy/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

RPNPY_SSMALL_NAME  = rpnpy$(RPNPY_SFX)_$(RPNPY_VERSION)_all
RPNPY_SSMARCH_NAME = rpnpy$(RPNPY_SFX)+$(COMP_ARCH)_$(RPNPY_VERSION)_$(SSMARCH)
RPNPY_SSMALL_FILES  = $(RPNPY_SSMALL_NAME).ssm
RPNPY_SSMARCH_FILES =  ## $(RPNPY_SSMARCH_NAME).ssm

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
   RPNPY_VERSION_X = test/
   SSM_TEST_INSTALL_RELDIR = test/
endif

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
RPNPY_SSM_BASE_DOM  = $(SSM_BASE)/ENV/d/$(RPNPY_VERSION_X)rpnpy
RPNPY_SSM_BASE_BNDL = $(SSM_BASE)/ENV/$(RPNPY_VERSION_X)rpnpy
RPNPY_INSTALL   = rpnpy_install
RPNPY_UNINSTALL = rpnpy_uninstall

DIRORIG_rpnpy := $(rpnpy)
# BUILDSSM = $(TMPDIR)/build-rpnpy
BUILDSSM = $(ROOT)/$(CONST_BUILDSSM)

.PHONY: 
rpnpy_ssm: rpnpy_ssm_all.ssm rpnpy_ssm_arch.ssm
rm_rpnpy_ssm: rm_rpnpy_ssm_all.ssm rm_rpnpy_ssm_all rm_rpnpy_ssm_arch.ssm rpnpy_ssm_arch_rm

rpnpy_ssm_all.ssm: $(RPNPY_SSMALL_FILES)
$(RPNPY_SSMALL_FILES): rpnpy_ssm_all rm_rpnpy_ssm_all.ssm $(SSM_DEPOT_DIR)/$(RPNPY_SSMALL_NAME).ssm
rm_rpnpy_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(RPNPY_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(RPNPY_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

rpnpy_ssm_all: rm_rpnpy_ssm_all $(BUILDSSM)/$(RPNPY_SSMALL_NAME)
rm_rpnpy_ssm_all:
	rm -rf $(BUILDSSM)/$(RPNPY_SSMALL_NAME)
$(BUILDSSM)/$(RPNPY_SSMALL_NAME): rpnpy_ssmusedep_bndl
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_rpnpy)/.ssm.d/exclude $(DIRORIG_rpnpy)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control rpnpy $(RPNPY_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

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
	mkdir -p $@/lib/$(EC_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/lib/libdummy_$(RPNPY_SSMARCH_NAME).a ; \
	cd $(LIBDIR) ; \
	rsync -av `ls librpnpy*.a librpnpy*.a.fl librpnpy*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	if [[ x$(MAKE_SSM_NOMOD) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
		touch $@/include/dummy_$(RPNPY_SSMARCH_NAME).inc ; \
		cd $(MODDIR) ; \
		cp $(RPNPY_MOD_FILES) $@/include/$(EC_ARCH) ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOINC) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		touch $@/include/dummy_$(RPNPY_SSMARCH_NAME).mod ; \
		echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
		.rdemkversionfile rpnpy $(RPNPY_VERSION) $@/include/$(EC_ARCH) f ; \
		.rdemkversionfile rpnpy $(RPNPY_VERSION) $@/include/$(EC_ARCH) c ; \
		.rdemkversionfile rpnpy $(RPNPY_VERSION) $@/include/$(EC_ARCH) sh ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOABS) != x1 ]] ; then \
		mkdir -p $@/bin/$(BASE_ARCH) ; \
		touch $@/bin/dummy_$(RPNPY_SSMARCH_NAME).bin ; \
		cd $(BINDIR) ; \
		cp $(RPNPY_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	fi ; \
	cp -R $(DIRORIG_rpnpy)/.ssm.d $@/ ; \
	.rdemk_ssm_control rpnpy $(RPNPY_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control


.PHONY: rpnpy_ssmusedep_bndl rpnpy_ssmusedep_bndl_rm rpnpy_ssmusedep_bndl_all
rpnpy_ssmusedep_bndl: | rpnpy_ssmusedep_bndl_rm rpnpy_ssmusedep_bndl_all
rpnpy_ssmusedep_bndl_rm:
	rm -f $(rpnpy)/ssmusedep.bndl $(rpnpy)/ssmusedep_post.bndl
rpnpy_ssmusedep_bndl_all: $(rpnpy)/ssmusedep.bndl $(rpnpy)/ssmusedep_post.bndl
	ls -l $(rpnpy)/ssmusedep.bndl $(rpnpy)/ssmusedep_post.bndl
$(rpnpy)/ssmusedep.bndl:
	touch $@
	# touch $@ ;\
	# if [[ -f $(rpnpy)/DEPENDENCIES.external.bndl ]] ; then \
	#    cat $(rpnpy)/DEPENDENCIES.external.bndl >> $@ ;\
	# fi ;\
	# echo >> $@ ;\
	# if [[ -f $(rpnpy)/DEPENDENCIES.external.$${RDENETWORK}.bndl ]] ; then \
	#    cat $(rpnpy)/DEPENDENCIES.external.$${RDENETWORK}.bndl >> $@ ;\
	# fi ;\
	# echo >> $@ ;\
	# if [[ -f $(rpnpy)/DEPENDENCIES.mig.bndl ]] ; then \
	#    for i in `cat $(rpnpy)/DEPENDENCIES.mig.bndl | tr '\n' ' '` ; do \
	#       i2="$$(echo $$i | sed 's|/x/|/|g')" ;\
	#       if [[ "x$(SSM_TEST_INSTALL_RELDIR)" == "x" ]] ; then i2=$$i ; fi ;\
	#       i0=$${i2%/*} ;\
	#       i1=`echo $${i2} | sed "s|$${i0%/*}/||"` ;\
	#       echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} >> $@ ;\
	#       echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} ;\
	#    done ;\
	# fi
$(rpnpy)/ssmusedep_post.bndl:
	touch $@

.PHONY: rpnpy_install rpnpy_uninstall
rpnpy_install: rpnpy_ssmusedep_bndl
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git $(SSM_SKIP_INSTALLED) \
			--dest=$(RPNPY_SSM_BASE_DOM)/rpnpy_$(RPNPY_VERSION) \
			--bndl=$(RPNPY_SSM_BASE_BNDL)/$(RPNPY_VERSION).bndl \
			--pre=$(rpnpy)/ssmusedep.bndl \
			--post=$(rpnpy)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			rpnpy{_,+*_,-d+*_}$(RPNPY_VERSION)_*.ssm

rpnpy_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(RPNPY_SSM_BASE_DOM)/rpnpy_$(RPNPY_VERSION) \
			--bndl=$(RPNPY_SSM_BASE_BNDL)/$(RPNPY_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnpy/include/Makefile.ssm.mk [END] ====================)
endif
