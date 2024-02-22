include ../../Makefile.defs

# This particular test will try to link a non-existent file and check for the
# expected error message using FileCheck.

# The TESTSRC_ALL does only contain a dummy program to allow automatic checking
# as well as returning a zero error-code.

TESTNAME     = link-missing-file
TESTSRC_ALL  = link-missing-file.c

# Since llvm-link would return a non-zero exit-code, use the dummy program to
# indicate 'success'.
RUNCMD       = $(AOMP)/bin/llvm-link MissingFile.bc 2>&1 | \
               $(AOMP)/bin/FileCheck $(TESTSRC_ALL); ./$(TESTNAME)

include ../Makefile.rules
