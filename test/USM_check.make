	include Makefile.defs

.SILENT:
.PHONY: all

SOUGHT_GPU := $(findstring $(GPU), $(SUPPORTS_USM))

all::
	if [ "$(SOUGHT_GPU)" ]; then echo "true"; else echo "false"; fi

