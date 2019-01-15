

OTHERMAKEFILE2 = $(wildcard $(OTHERMAKEFILE))
ifneq (,$(OTHERMAKEFILE2))
include $(OTHERMAKEFILE2)
endif

print1-%:
	@echo $* = $($*)

print-%:
	@echo $($*)
