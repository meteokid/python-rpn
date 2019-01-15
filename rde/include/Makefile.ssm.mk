ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: Makefile.ssm.mk)
$(info ## )
endif

ifeq (,$(HOME))
   $(error FATAL ERROR: HOME is not defined)
endif

## ==== Pkg Building Macros

BUILDSSM    := $(ROOT)/$(shell rdevar build/ssm)
SSMARCH_OLD := $(shell .rdessmarch)
SSMORDARCH  := $(shell .rdessmarch --ord)
SSMARCH     = $(SSMORDARCH)

SSMDEPOTDIR = $(HOME)/SsmDepot
# export SSMTMPLDIR = $(rde)/share/ssm_tmpl
# export SSMTMPLNAMEARCH  := PKG_V998V_ARCH

## e.g: $(eval $(call SSMARCH_template,modelutils,MODELUTILS,modelutils+$(COMP_ARCH)_$(MODELUTILS_VERSION)_$(SSMARCH)))
# SSMARCH_template = \
# $$(BUILD)/$(3): ; \
# mkdir -p $$@/include/$$(EC_ARCH) $$@/lib/$$(EC_ARCH) $$@/bin/$$(BASE_ARCH) ; \
# cp $$($(2)_MOD_FILES) $$@/include/$$(EC_ARCH) ; \
# rsync -av $$(wildcard lib$(1)*.a lib$(1)*.a.fl lib$(1)*.so) $$@/lib/$$(EC_ARCH)/ ; \
# .rdemkversionfile $(1) $$($(2)_VERSION) $$@/include/$$(EC_ARCH) f ; \
# .rdemkversionfile $(1) $$($(2)_VERSION) $$@/include/$$(EC_ARCH) c ; \
# .rdemkversionfile $(1) $$($(2)_VERSION) $$@/include/$$(EC_ARCH) sh ; \
# cd $$(BINDIR) ; \
# cp $$($(2)_ABS_FILES) $$@/bin/$$(BASE_ARCH) ; \
# cp -R $$(DIRORIG_$(1))/.ssm.d $$@/ ; \
# .rdemk_ssm_control modelutils $$($(2)_VERSION) "$$(SSMORDARCH) ; $$(SSMARCH_OLD) ; $$(BASE_ARCH)" $$@/BUILDINFO $$@/DESCRIPTION > $$@/.ssm.d/control ; \

# ## e.g: $(eval $(call SSMALL_template,modelutils,MODELUTILS,modelutils_$(MODELUTILS_VERSION)_all))
# SSMALL_template = \
# $$(BUILD)/$(3): ; \
# rm -rf $$@ ; mkdir -p $$@ ; \
# rsync -av --exclude-from=$$(DIRORIG_$(1))/.ssm.d/exclude $$(DIRORIG_$(1))/ $$@/ ; \
# echo "Dependencies (s.ssmuse.dot): " > $$@/BUILDINFO ; \
# cat $$@/ssmusedep.bndl >> $$@/BUILDINFO ; \
# .rdemk_ssm_control $(1) $$($(2)_VERSION) "all ; $$(BASE_ARCH)" $$@/BUILDINFO $$@/DESCRIPTION > $$@/.ssm.d/control


# BUILDINFO:
# 	echo "Dependencies (s.ssmuse.dot): " > $@
# 	cat ssmuse_dependencies.bndl >> $@
# $(DISTDIRARCH)/$(PKG_VER_ALL): | $(DISTDIRARCH) $(DISTDIRROOTLNK) $(BUILDINFO)
# 	(cd $(DISTDIRARCH) ;\
# 	$(TAR) xzf $(SSMTMPLDIR)/$(SSMTMPLNAMEARCH).ssm ;\
# 	mv -f $(SSMTMPLNAMEARCH) $@ 2>/dev/null || true) ;\
# 	echo $(shell mu.mk_ssm_control $(PKGNAME) $(VERSION) "all ; $(BASE_ARCH)" $(TOPDIR)/BUILDINFO $(TOPDIR)/DESCRIPTION > $(DISTDIRARCH)/SSMCONTROLFILE_all) ;\
# 	mv -f $(DISTDIRARCH)/SSMCONTROLFILE_all $@/.ssm.d/control ;\
# 	for mydir in $(SUBDIRS_SSMALL) ; do \
# 		$(MAKE) -f $(modelutils)/include/Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $@/$${mydir} ;\
# 		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
# 	done ;\
# 	mkdir -p $@/lib 2>/dev/null || true ;
# 	touch $@/lib/libdummy.a 2>/dev/null || true
# 	#touch is a patch to force smm to add $@/include to EC_INCLUDE_PATH
# $(DISTDIRARCH)/$(PKG_VER_ARCH): | $(DISTDIRARCH) $(BUILDINFO)
# 	(cd $(DISTDIRARCH) ;\
# 	$(TAR) xzf $(SSMTMPLDIR)/$(SSMTMPLNAMEARCH).ssm ;\
# 	mv -f $(SSMTMPLNAMEARCH) $@ 2>/dev/null || true) ;\
# 	echo $(shell mu.mk_ssm_control $(PKGNAME) $(VERSION) "$(SSMARCH) ; $(BASE_ARCH)" $(TOPDIR)/BUILDINFO $(TOPDIR)/DESCRIPTION > $(DISTDIRARCH)/SSMCONTROLFILE_$(SSMARCH)) ;\
# 	mv -f $(DISTDIRARCH)/SSMCONTROLFILE_$(SSMARCH) $@/.ssm.d/control ;\
# 	for mydir in $(SUBDIRS_SSMARCH) ; do \
# 		$(MAKE) -f $(modelutils)/include/Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $@/$${mydir} ;\
# 		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
# 	done 
# $(DISTDIRARCH)/$(PKG_VER_ARCH)/include/$(COMP_ARCH): | $(DISTDIRARCH)/$(PKG_VER_ARCH)/include/$(EC_ARCH)
# 	ln -s ./$(EC_ARCH)/. $@ 2>/dev/null || true
# $(DISTDIRARCH)/$(PKG_VER_ARCH)/include/$(EC_ARCH): objects
# 	mkdir -p $@ 2>/dev/null || true ;\
# 	cp $(BUILDDIRARCH)/*.mod $(BUILDDIRARCH)/*/*.mod $@/ 2>/dev/null || true ;\
# 	cp $(BUILDDIRARCH)/include/$(PKGNAME)_version.* $@/ 2>/dev/null || true
# $(DISTDIRARCH)/$(PKG_VER_ARCH)/lib/$(COMP_ARCH): | $(DISTDIRARCH)/$(PKG_VER_ARCH)/lib/$(EC_ARCH)
# 	ln -s ./$(EC_ARCH)/. $@ 2>/dev/null || true
# $(DISTDIRARCH)/$(PKG_VER_ARCH)/lib/$(EC_ARCH): libs libs_shared
# 	mkdir -p $@ 2>/dev/null || true ;\
# 	rsync -a $(BUILDDIRARCH)/lib*.a $(BUILDDIRARCH)/*/lib*.a $(BUILDDIRARCH)/lib*.a.fl $(BUILDDIRARCH)/*/lib*.a.fl $(BUILDDIRARCH)/lib*.so $(BUILDDIRARCH)/*/lib*.so $@/ 2>/dev/null || true ;\
# 	cp $(BUILDDIRARCH)/.VERSION $@/ 2>/dev/null || true

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.ssm.mk [END] =========================================)
endif
