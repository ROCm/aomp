include ../Makefile.defs

TESTS_DIR = \
  SAtealeaf \
  aomp-issue376 \
  aomp-issue531 \
  link-missing-file \
  multi-sdma \
  no-loop-1 \
  no-loop-2 \
  no-loop-3 \
  no-loop-4 \
  no-loop-5 \
  no-loop-6 \
  no-loop-7 \
  no-loop-threads \
  omptest-device-emi \
  omptest-device-non-emi \
  opt-kernel-global-iv \
  opt-kernel-inner-pragma-1 \
  opt_kerns_clauses \
  usm-locals-pragma-xnack-disabled-xnack-plus \
  usm-locals-pragma-xnack-enabled-xnack-any \
  usm-locals-pragma-xnack-enabled-xnack-minus \
  usm-locals-pragma-xnack-enabled-xnack-plus \
  usm-locals-xnack-disabled-xnack-any \
  usm-locals-xnack-disabled-xnack-any-apu-maps \
  usm-locals-xnack-disabled-xnack-minus \
  usm-locals-xnack-disabled-xnack-plus \
  usm-locals-xnack-enabled-xnack-any-apu \
  usm-locals-xnack-enabled-xnack-any-apu-maps-non-apu \
  usm-locals-xnack-enabled-xnack-any-non-apu \
  usm-locals-xnack-enabled-xnack-minus \
  usm-locals-xnack-enabled-xnack-plus-apu \
  usm-locals-xnack-enabled-xnack-plus-apu-maps-non-apu \
  usm-locals-xnack-enabled-xnack-plus-non-apu \
  veccopy-ctor-1 \
  veccopy-ompt-ctor-1 \
  veccopy-ompt-target \
  veccopy-ompt-target-cmake \
  veccopy-ompt-target-data-emi \
  veccopy-ompt-target-data-tracing-emi \
  veccopy-ompt-target-default-device \
  veccopy-ompt-target-devices \
  veccopy-ompt-target-emi \
  veccopy-ompt-target-emi-map \
  veccopy-ompt-target-emi-tracing \
  veccopy-ompt-target-emi-tracing-sync-async \
  veccopy-ompt-target-emi-tracing-sync-force \
  veccopy-ompt-target-emi-tracing-sync-size \
  veccopy-ompt-target-map \
  veccopy-ompt-target-tracing \
  veccopy-ompt-target-translate-time \
  veccopy-ompt-target-type-device-time \
  workgroup_size_option1 \
  workgroup_size_option2 \
  xteam-red-fast-1 \
  xteam-red-split-1 \
  xteam-red-split-2 \

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
