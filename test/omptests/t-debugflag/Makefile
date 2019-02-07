include ../Makefile.defs

EXTRAFLAGS := -O0 -g

# Overwrite the env var FORCE_OPT which will set the -O3 flag in Makefile.rules.
# This test cannot use it because ptxas does not support optimized debugging and
# compiling the test will fail.
FORCE_OPT :=

################################################################################
#
# Include any special rule for this testcase here! Usually:
#
# static:
#
# dynamic:
#
################################################################################

include ../Makefile.rules

