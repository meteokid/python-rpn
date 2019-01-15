SHELL = /bin/bash

MODELSCRIPTS_VERSION0  = x/5.0.rc1
MODELSCRIPTS_VERSION   = $(notdir $(MODELSCRIPTS_VERSION0))
MODELSCRIPTS_VERSION_X = $(dir $(MODELSCRIPTS_VERSION0))

MODELSCRIPTS_SSMALL_NAME  = modelscripts_$(MODELSCRIPTS_VERSION)_all
MODELSCRIPTS_SSMALL_FILES  = $(MODELSCRIPTS_SSMALL_NAME).ssm

SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE2     := $(HOME)/SsmBundles
MODELSCRIPTS_SSM_BASE_DOM  = $(SSM_BASE)/GEM/d/$(MODELSCRIPTS_VERSION_X)modelscripts
MODELSCRIPTS_SSM_BASE_BNDL = $(SSM_BASE)/GEM/$(MODELSCRIPTS_VERSION_X)modelscripts
MODELSCRIPTS_INSTALL   = modelscripts_install
MODELSCRIPTS_UNINSTALL = modelscripts_uninstall

DIRORIG_modelscripts := $(PWD)
BUILDSSM = $(TMPDIR)/build-modelscripts

ssm: $(MODELSCRIPTS_SSMALL_FILES)
modelscripts_ssm_all.ssm: $(MODELSCRIPTS_SSMALL_FILES)
$(MODELSCRIPTS_SSMALL_FILES): modelscripts_ssm_all rm_modelscripts_ssm_all.ssm $(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMALL_NAME).ssm
rm_modelscripts_ssm_all.ssm:
	rm -f $(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMALL_NAME).ssm
$(SSM_DEPOT_DIR)/$(MODELSCRIPTS_SSMALL_NAME).ssm:
	cd $(BUILDSSM) ;\
	chmod a+x $(basename $(notdir $@))/bin/* 2>/dev/null || true ;\
	tar czvf $@ $(basename $(notdir $@))
	ls -l $@

modelscripts_ssm_all: rm_modelscripts_ssm_all $(BUILDSSM)/$(MODELSCRIPTS_SSMALL_NAME)
rm_modelscripts_ssm_all:
	rm -rf $(BUILDSSM)/$(MODELSCRIPTS_SSMALL_NAME)
$(BUILDSSM)/$(MODELSCRIPTS_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_modelscripts)/.ssm.d/exclude $(DIRORIG_modelscripts)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	echo "CURRENT_VERSION=$(MODELSCRIPTS_VERSION)" > $@/share/gem-mod/mod/VERSION ; \
	.rdemk_ssm_control modelscripts $(MODELSCRIPTS_VERSION) all $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

.PHONY: install uninstall modelscripts_install modelscripts_uninstall
install: modelscripts_install
#TODO: install all pkg should be a git repos
modelscripts_install: 
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--git \
			--dest=$(MODELSCRIPTS_SSM_BASE_DOM)/modelscripts_$(MODELSCRIPTS_VERSION) \
			--bndl=$(MODELSCRIPTS_SSM_BASE_BNDL)/$(MODELSCRIPTS_VERSION).bndl \
			--pre=$(DIRORIG_modelscripts)/ssmusedep.bndl \
			--base=$(SSM_BASE2) \
			modelscripts{_,+*_}$(MODELSCRIPTS_VERSION)_*.ssm

uninstall: modelscripts_uninstall
modelscripts_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v \
			--dest=$(MODELSCRIPTS_SSM_BASE_DOM)/modelscripts_$(MODELSCRIPTS_VERSION) \
			--bndl=$(MODELSCRIPTS_SSM_BASE_BNDL)/$(MODELSCRIPTS_VERSION).bndl \
			--base=$(SSM_BASE2) \
			--uninstall
