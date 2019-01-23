ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rde/include/Makefile.ssm.mk)
$(info ## )
endif

#------------------------------------

RDE_SSMALL_NAME  = rde$(RDE_SFX)_$(RDE_VERSION)_all
RDE_SSMARCH_NAME = rde$(RDE_SFX)+$(COMP_ARCH)_$(RDE_VERSION)_$(SSMARCH)
RDE_SSMALL_FILES  = $(RDE_SSMALL_NAME).ssm
RDE_SSMARCH_FILES =  ## $(RDE_SSMARCH_NAME).ssm

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
   RDE_VERSION_X = test/
   SSM_TEST_INSTALL_RELDIR = test/
endif

SSMORDARCH    := $(ORDENV_PLAT)
SSMARCH        = $(SSMORDARCH)

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
RDE_SSM_BASE_DOM  = $(SSM_BASE)/ENV/d/$(RDE_VERSION_X)rde
RDE_SSM_BASE_BNDL = $(SSM_BASE)/ENV/$(RDE_VERSION_X)rde
RDE_INSTALL   = rde_install
RDE_UNINSTALL = rde_uninstall

DIRORIG_rde := $(rde)
# BUILDSSM = $(TMPDIR)/build-rde
BUILDSSM = $(ROOT)/$(CONST_BUILDSSM)


ifeq (,$(RDE_MAKEFILE_SSM_INCLUDED))

.PHONY: 
rde_ssm: rde_ssm_all.ssm rde_ssm_arch.ssm
rm_rde_ssm: rm_rde_ssm_all.ssm rm_rde_ssm_all rm_rde_ssm_arch.ssm rde_ssm_arch_rm

rde_ssm_all.ssm: $(RDE_SSMALL_FILES)
$(RDE_SSMALL_FILES): rde_ssm_all rm_rde_ssm_all.ssm $(SSM_DEPOT_DIR)/$(RDE_SSMALL_NAME).ssm
rm_rde_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(RDE_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(RDE_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

rde_ssm_all: rm_rde_ssm_all $(BUILDSSM)/$(RDE_SSMALL_NAME)
rm_rde_ssm_all:
	rm -rf $(BUILDSSM)/$(RDE_SSMALL_NAME)
$(BUILDSSM)/$(RDE_SSMALL_NAME): rde_ssmusedep_bndl
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_rde)/.ssm.d/exclude $(DIRORIG_rde)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control rde $(RDE_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

rde_ssm_arch.ssm: $(RDE_SSMARCH_FILES)
$(RDE_SSMARCH_FILES): rde_ssm_arch rm_rde_ssm_arch.ssm $(SSM_DEPOT_DIR)/$(RDE_SSMARCH_NAME).ssm
rm_rde_ssm_arch.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(RDE_SSMARCH_NAME).ssm
$(SSM_DEPOT_DIR)/$(RDE_SSMARCH_NAME).ssm:
	cd $(BUILDSSM) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

rde_ssm_arch: rde_ssm_arch_rm $(BUILDSSM)/$(RDE_SSMARCH_NAME)
rde_ssm_arch_rm:
	rm -rf $(BUILDSSM)/$(RDE_SSMARCH_NAME)
$(BUILDSSM)/$(RDE_SSMARCH_NAME):
	mkdir -p $@/lib/$(EC_ARCH) ; \
	ln -s ./$(EC_ARCH)/. $@/lib/$(COMP_ARCH) ; \
	touch $@/lib/libdummy_$(RDE_SSMARCH_NAME).a ; \
	cd $(LIBDIR) ; \
	rsync -av `ls librde*.a librde*.a.fl librde*.so 2>/dev/null` $@/lib/$(EC_ARCH)/ ; \
	if [[ x$(MAKE_SSM_NOMOD) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		ln -s ./$(EC_ARCH)/. $@/include/$(COMP_ARCH) ; \
		touch $@/include/dummy_$(RDE_SSMARCH_NAME).inc ; \
		cd $(MODDIR) ; \
		cp $(RDE_MOD_FILES) $@/include/$(EC_ARCH) ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOINC) != x1 ]] ; then \
		mkdir -p $@/include/$(EC_ARCH) ; \
		touch $@/include/dummy_$(RDE_SSMARCH_NAME).mod ; \
		echo $(BASE_ARCH) > $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT) >> $@/include/$(BASE_ARCH)/.restricted ; \
		echo $(EC_ARCH) > $@/include/$(EC_ARCH)/.restricted ; \
		echo $(ORDENV_PLAT)/$(COMP_ARCH) >> $@/include/$(EC_ARCH)/.restricted ; \
		.rdemkversionfile rde $(RDE_VERSION) $@/include/$(EC_ARCH) f ; \
		.rdemkversionfile rde $(RDE_VERSION) $@/include/$(EC_ARCH) c ; \
		.rdemkversionfile rde $(RDE_VERSION) $@/include/$(EC_ARCH) sh ; \
	fi ; \
	if [[ x$(MAKE_SSM_NOABS) != x1 ]] ; then \
		mkdir -p $@/bin/$(BASE_ARCH) ; \
		touch $@/bin/dummy_$(RDE_SSMARCH_NAME).bin ; \
		cd $(BINDIR) ; \
		cp $(RDE_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	fi ; \
	cp -R $(DIRORIG_rde)/.ssm.d $@/ ; \
	.rdemk_ssm_control rde $(RDE_VERSION) $(SSMORDARCH) $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control


.PHONY: rde_ssmusedep_bndl rde_ssmusedep_bndl_rm rde_ssmusedep_bndl_all
rde_ssmusedep_bndl: | rde_ssmusedep_bndl_rm rde_ssmusedep_bndl_all
rde_ssmusedep_bndl_rm:
	rm -f $(rde)/ssmusedep.bndl $(rde)/ssmusedep_post.bndl
rde_ssmusedep_bndl_all: $(rde)/ssmusedep.bndl $(rde)/ssmusedep_post.bndl
	ls -l $(rde)/ssmusedep.bndl $(rde)/ssmusedep_post.bndl
$(rde)/ssmusedep.bndl:
	touch $@ ;\
	if [[ -f $(rde)/DEPENDENCIES.external.bndl ]] ; then \
	   cat $(rde)/DEPENDENCIES.external.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(rde)/DEPENDENCIES.external.$${RDENETWORK}.bndl ]] ; then \
	   cat $(rde)/DEPENDENCIES.external.$${RDENETWORK}.bndl >> $@ ;\
	fi ;\
	echo >> $@ ;\
	if [[ -f $(rde)/DEPENDENCIES.mig.bndl ]] ; then \
	   for i in `cat $(rde)/DEPENDENCIES.mig.bndl | tr '\n' ' '` ; do \
	      i2="$$(echo $$i | sed 's|/x/|/|g')" ;\
	      if [[ "x$(SSM_TEST_INSTALL_RELDIR)" == "x" ]] ; then i2=$$i ; fi ;\
	      i0=$${i2%/*} ;\
	      i1=`echo $${i2} | sed "s|$${i0%/*}/||"` ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} >> $@ ;\
	      echo $(SSM_PREFIX)$${i0%/*}/$(SSM_TEST_INSTALL_RELDIR)$${i1} ;\
	   done ;\
	fi
$(rde)/ssmusedep_post.bndl:
	touch $@

.PHONY: rde_install rde_uninstall
rde_install: rde_ssmusedep_bndl
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git $(SSM_SKIP_INSTALLED) \
			--dest=$(RDE_SSM_BASE_DOM)/rde_$(RDE_VERSION) \
			--bndl=$(RDE_SSM_BASE_BNDL)/$(RDE_VERSION).bndl \
			--pre=$(rde)/ssmusedep.bndl \
			--post=$(rde)/ssmusedep_post.bndl \
			--base=$(SSM_BASE2) \
			rde{_,+*_,-d+*_}$(RDE_VERSION)_*.ssm

rde_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(RDE_SSM_BASE_DOM)/rde_$(RDE_VERSION) \
			--bndl=$(RDE_SSM_BASE_BNDL)/$(RDE_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall

endif #RDE_MAKEFILE_SSM_INCLUDED
RDE_MAKEFILE_SSM_INCLUDED := 1

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rde/include/Makefile.ssm.mk [END] ====================)
endif
