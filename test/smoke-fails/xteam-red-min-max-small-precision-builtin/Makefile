include ../../Makefile.defs

TESTNAME     = xteam-red-min-max-small-precision-builtin
TESTSRC_MAIN = xteam-red-min-max-small-precision-builtin.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ifneq ("$(wildcard $(AOMP)/lib/libmlir_float16_utils.so)","")
  CFLAGS += -lmlir_float16_utils
endif

EXTRA_LDFLAGS     += -lm
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
