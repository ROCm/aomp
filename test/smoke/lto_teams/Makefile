include ../../Makefile.defs

TESTNAME     = lto_teams
TESTSRC_MAIN = lto_teams.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

# NOTE: It is possible to test this with ROCM compiler by setting these two variables
#    export AOMP=/opt/rocm/llvm
#    export AOMP_GPU=`/opt/rocm/aomp/bin/mygpu`

_aocc_default = /opt/AMD/aocc-compiler-3.0.0
ifeq ($(shell ls $(_aocc_default)/bin/ld.lld 2>/dev/null),)
  _aocc_default=$(AOMP)
endif
AOCC ?= $(_aocc_default)
_aocc_lld = $(shell ls $(AOCC)/bin/ld.lld 2>/dev/null)
ifeq ($(_aocc_lld),)
  $(info  )
  $(info ERROR Cannot find alternate linker $(AOCC)/bin/ld.lld )
  $(info       Check value of AOCC environment variable. AOCC=$(AOCC))
  $(info  )
  $(error Alternate linker $(AOCC)/bin/ld.lld does not exist)
endif
_aocc_llvm_version=$(shell $(_aocc_lld) --version | grep "LLD" | cut -d" " -f2 | cut -d"." -f1)
_aomp_llvm_version=$(shell $(AOMP)/bin/ld.lld --version | grep "LLD" | cut -d" " -f2 | cut -d"." -f1)
ifneq ($(_aomp_llvm_version),$(_aocc_llvm_version))
   $(info )
   $(info WARNING LLVM version $(_aomp_llvm_version) is different that AOCC LLVM version $(_aocc_llvm_version) )
   $(info         Alternate linker is $(_aocc_lld) )
   $(info )
endif
# The alternate linker flag changed to --ld-path in LLVM 12
ifeq ($(_aomp_llvm_version),11)
   _ldflag=-fuse-ld=
else
   _ldflag=--ld-path=
endif

#  This fails when gnu linker does not have LLVM linker plugin
#ALT_LINKER =
#  This option works because it the llvm linker that is part of AOMP installation
#ALT_LINKER = $(ld_flag)lld
#  This option uses AOCC as the alternative linker called by AOMP driver
ALT_LINKER = $(_ldflag)$(_aocc_lld)

CFLAGS	   = -O3 -flto=thin $(ALT_LINKER)
CLANG      ?= clang
OMP_BIN    = $(AOMP)/bin/$(CLANG)
CC         = $(OMP_BIN) $(VERBOSE)

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules

