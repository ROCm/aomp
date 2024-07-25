include ../Makefile.defs

TESTS_DIR = \
    1D-full-bounds-exp-map \
    1D-partial-bounds-exp-map \
    bycopy-implicit-map \
    declare-target-and-explicit-map \
    declare-target-and-implicit-and-explicit-map \
    declare-target-link-1 \
    declare-target-link-2 \
    declare-target-link-3 \
    exp-real-map \
    fdefault-sizeof \
    fdefault-type-8 \
    flang-274768 \
    flang-274768-1 \
    flang-404247 \
    flang-416048 \
    flang-420573 \
    flang-421111-1 \
    flang-421111-2 \
    flang-421353 \
    flang-439387 \
    flang-439387a \
    flang-439387b \
    flang-439387c \
    flang-440121 \
    flang-441477 \
    flang-443915 \
    flang-445718 \
    flang-445718-fortran-intrinsic \
    flang-445882 \
    flang-445888 \
    flang-446057a \
    flang-446057b \
    flang-446057c \
    flang-446057d \
    flang-446525 \
    flang-453308 \
    flang-453308a \
    flang-446005a \
    flang-446005b \
    flang-446033c_ttdpd \
    flang-446033d_ttdpd \
    flang-446034 \
    flang-445718-sign \
    flang-457354 \
    flang-460740 \
    flang-462537 \
    flang-463690 \
    flang-471469 \
    flang-472452 \
    flang-use-device \
    intrin-rename-func \
    intrin-rename-sub \
    LaunchLatency \
    llvm-issue-82047 \
    mathdbl \
    mathquad \
    milestone-1 \
    milestone-2 \
    milestone-3-babel \
    milestone-3-babel-drop-1 \
    milestone-3-babel-noteams \
    mix-host-target \
    olcf-tc-0-serial \
    olcf-tc-1-openmp-cpu \
    olcf-tc-2-openmp-gpu-teams \
    olcf-tc-3-openmp-gpu-parallel \
    olcf-tc-4-openmp-gpu-data \
    olcf-tc-5-openmp-gpu-implicit \
    olcf-tc-6-openmp-combined \
    olcf-tc-7-loop-combined \
    olcf-tf-0-serial \
    olcf-tf-1-openmp-cpu \
    olcf-tf-2-openmp-gpu-teams \
    olcf-tf-3-openmp-gpu-teams-parallel \
    olcf-tf-4-openmp-gpu-teams-parallel-data \
    olcf-tf-5-openmp-gpu-teams-parallel-implicit-map \
    olcf-tf-6-combined \
    single-value-map \
    target-cray-pointer \
    teams-distribute-no-parallel \
    teams-ditribute-collapse \
    vmuln

all:
	@for test_dir in $(TESTS_DIR); do \
	  echo; \
	  test_name=`grep "TESTNAME *=" $$test_dir/Makefile | sed "s/.*= *//"`; \
	  echo "TEST_DIR: $$test_dir\tTEST_NAME: $$test_name\tMAKE: $(MAKE) -C $$test_dir"; \
	  $(MAKE) -C $$test_dir; \
	done

run run_obin run_sbin run_llbin clean clean_log llbin sbin obin:
	@for test_dir in $(TESTS_DIR); do \
	  echo $$nnn; \
	  test_name=`grep "TESTNAME *=" $$test_dir/Makefile | sed "s/.*= *//"`; \
	  echo "TEST_DIR: $$test_dir\tTEST_NAME: $$test_name\tMAKE: $(MAKE) -C $$test_dir $@"; \
	  $(MAKE) -C $$test_dir $@; \
	done

check:
	 @for test_dir in $(TESTS_DIR); do \
          echo $$nnn; \
          test_name=`grep "TESTNAME *=" $$test_dir/Makefile | sed "s/.*= *//"`; \
          echo "TEST_DIR: $$test_dir\tTEST_NAME: $$test_name\tMAKE: $(MAKE) -C $$test_dir $@"; \
          $(MAKE) -C $$test_dir $@; \
        done

.ll .ll.s .ll.o .s .s.o .o:
	@for test_dir in $(TESTS_DIR); do \
	  echo $$nnn; \
	  test_name=`grep "TESTNAME *=" $$test_dir/Makefile | sed "s/.*= *//"`; \
	  echo "TEST_DIR: $$test_dir\tTEST_NAME: $$test_name\tMAKE: $(MAKE) -C $$test_dir $$test_name$@"; \
	  $(MAKE) -C $$test_dir $$test_name$@; \
	done

help:
	@echo
	@echo "LLVM Tool Chain:	$(AOMP)/bin"
	@echo "Offload Targets:	$(TARGET)"
	@echo "Host Target:		$(AOMP_CPUTARGET)"
	@echo "Application Dirs:	$(TESTS_DIR)"
	@echo
	@echo "This Makefile supports the following flags:"
	@echo
	@echo "	make llbin      // Link pass only"
	@echo "	make run_llbin  // Execute llbin"
	@echo
	@echo "	make sbin       // Link pass only"
	@echo "	make run_sbin   // Execute sbin"
	@echo
	@echo "	make obin       // Link pass only"
	@echo "	make run_obin   // Execute obin"
	@echo
	@echo "	make .ll        // Compile pass only             : -c -S -emit-llvm"
	@echo "	make .ll.s      // Backend pass only             : -c -S"
	@echo "	make .ll.o      // Assemble pass only            : -c"
	@echo "	make .s         // Compile & Backend passes      : -c -S"
	@echo "	make .s.o       // Assemble pass only            : -c"
	@echo "	make .o         // Compile, Backend, Assemble    : -c"
	@echo
	@echo "	make            // All passes, build all examples from Application Dirs"
	@echo "	make run        // Execute all binaries from Application Dirs"
	@echo
	@echo "	make clean"
	@echo "	make clean_log"
	@echo "	make help"
	@echo
	@echo " Environment variables to control compilation & execution"
	@echo "    VERBOSE=1        See lots of compiler messages and driver actions"
	@echo "    TEMPS=1          Do not delete intermediate files"
	@echo "    OFFLOAD_DEBUG=1  See Runtime diagnostics for each call to libomptarget API"
	@echo "    TARGET=          Override Makefile target"
	@echo
	@echo " Compile Environment:  $(SETENV)"
	@echo
	@echo " Run Environment:      $(RUNENV)"
	@echo
	@echo " Compile Flags:        $(CFLAGS) $(EXTRA_CFLAGS)"
	@echo
	@echo " OMP Compile Flags:    $(OMP_FLAGS) $(EXTRA_OMP_FLAGS)"
	@echo
	@echo " Link Flags:           $(LINK_FLAGS) $(EXTRA_LDFLAGS)"
	@echo
