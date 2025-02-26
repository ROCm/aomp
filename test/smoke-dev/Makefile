include ../Makefile.defs

TESTS_DIR = \
    aomp_mappings_newrt \
    check-xnack \
    clang-ifaces \
    fast_fp_atomics_mi200_default_usm_map \
    fast_fp_atomics_mi200_enable_corase_grain_usm_map \
    fast_fp_atomics_mi200_not_enable_corase_grain_usm_map \
    flang-280342 \
    flang_deviceio \
    flang-f90print \
    flang-host-targ \
    flang-loop0 \
    flang-TL-0 \
    flang-TL-1 \
    flang-TP-0 \
    flang-TP-1 \
    global-usm-fort \
    gpurun-l-m \
    hipreg_usm \
    launch_latency \
    lib-debug \
    mix_hipmemset_omp \
    occupancy-based-opt \
    occupancy-based-opt-big-jump-loop \
    occupancy-based-opt-xteam-reduction \
    omp_lock \
    red_bug_51 \
    reduction_team \
    reduction_teams \
    rt-tuning-envar \
    snap_red \
    swdev-461650 \
    test-offload-profile-envar \
    usm-locals-pragma-xnack-disabled-xnack-any \
    usm-locals-pragma-xnack-disabled-xnack-minus \
    veccopy-ompt-target-emi-tracing-dag \
    vmulsum-hsa-stats \
    xteam-red-4 \
    xteam-red-callee \
    xteam-red-callee-default-option \
    xteam-red-callee-ptr \
    xteam-red-cmdline-default-option \
    xteam-red-collapse \
    xteam-red-collapse-default-option \
    xteam-red-default-option-1 \
    xteam-red-default-option-1-g \
    xteam-red-fast-default-option \
    xteam-red-incr \
    xteam-red-incr-default-option \
    xteam-red-sched-default-option \
    xteam-red-small-precision \
    xteam-red-split-collapse-default-option \
    xteam-red-split-default-option-1 \
    xteam-red-split-default-option-2 \
    xteam-red-teams-default-option \
    xteam-red-threads-default-option \
    xteam-scan-1 \
    xteam-scan-no-loop \
    zero-copy-xnack-enable-dgpu-map-disable-apu \
    zero-copy-xnack-enable-dgpu-map-enable-apu 

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
