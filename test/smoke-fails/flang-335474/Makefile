include ../../Makefile.defs

# This particular test will solely use a bash script.
# Hence, we build a dummy target to remain compliant.
TESTNAME     = flang-335474
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
DUMMYSRC     = echo 'int main(void) { return 0; }'

CLANG        = $(AOMP)/bin/clang
CC           = $(DUMMYSRC) | $(CLANG) -x c -

RUNCMD       = ./cmd.sh

include ../Makefile.rules
