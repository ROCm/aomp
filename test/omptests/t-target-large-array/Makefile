include ../Makefile.defs

#option with nvidia for proper test
EXTRAFLAGS := -O3 -I/usr/local/cuda/include -L /usr/local/cuda/lib64/ -lcudart -DUSE_PINNED=1

# option if no nvidia avail
#EXTRAFLAGS := -O3 -DUSE_PINNED=0

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
