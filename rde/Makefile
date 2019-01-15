SHELL = /bin/bash

RDE_VERSION0  = 1.0.6
RDE_VERSION   = $(notdir $(RDE_VERSION0))
RDE_VERSION_X = $(dir $(RDE_VERSION0))

RDE_SSMALL_NAME  = rde_$(RDE_VERSION)_all
RDE_SSMALL_FILES  = $(RDE_SSMALL_NAME).ssm

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
RDE_SSM_BASE_DOM  = $(SSM_BASE)/ENV/d/$(RDE_VERSION_X)rde
RDE_SSM_BASE_BNDL = $(SSM_BASE)/ENV/$(RDE_VERSION_X)rde
RDE_INSTALL   = rde_install
RDE_UNINSTALL = rde_uninstall

DIRORIG_rde := $(PWD)
BUILDSSM = $(TMPDIR)/build-rde

RDEMK_SSM_CONTROL=.rdemk_ssm_control
ifneq (,$(wildcard $(PWD)/bin/.rdemk_ssm_control))
RDEMK_SSM_CONTROL := $(PWD)/bin/.rdemk_ssm_control
endif
RDE_SSM_INSLALL=rdessm-install
ifneq (,$(wildcard $(PWD)/bin/rdessm-install))
RDE_SSM_INSLALL := $(PWD)/bin/rdessm-install
endif

ssm: $(RDE_SSMALL_FILES)
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
$(BUILDSSM)/$(RDE_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_rde)/.ssm.d/exclude $(DIRORIG_rde)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	$(RDEMK_SSM_CONTROL) rde $(RDE_VERSION) "all" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

.PHONY: install uninstall rde_install rde_uninstall
install: rde_install
rde_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	$(RDE_SSM_INSLALL) -v \
			--dest=$(RDE_SSM_BASE_DOM)/rde_$(RDE_VERSION) \
			--bndl=$(RDE_SSM_BASE_BNDL)/$(RDE_VERSION).bndl \
			--base=$(SSM_BASE2) \
			rde{_,+*_}$(RDE_VERSION)_*.ssm

uninstall: rde_uninstall
rde_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	$(RDE_SSM_INSLALL) -v \
			--dest=$(RDE_SSM_BASE_DOM)/rde_$(RDE_VERSION) \
			--bndl=$(RDE_SSM_BASE_BNDL)/$(RDE_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall
