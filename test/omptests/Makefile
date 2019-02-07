
#
# Check if some required definitions were made by the user.
#

ifeq ($(HOSTRTL),)
  $(error HOSTRTL is not set)
endif
ifeq ($(TARGETRTL),)
  $(error TARGETRTL is not set)
endif
ifeq ($(GLOMPRTL),)
  $(error GLOMPRTL is not set)
endif
ifeq ($(LLVMBIN),)
  $(error LLVMBIN is not set)
endif

export HOSTRTL
export TARGETRTL
export GLOMPRTL
export LLVMBIN
export OMPTESTS_CUDA_COMPUTE_CAPABILITY

#
# Look for the testcase folders 
#

ifeq ($(TEST_FOLDERS),)
	machine := $(shell uname -m)

  TEST_FOLDERS := $(wildcard t-*)

  # Check if we are using LOMP - if so, add special tests for it.
  ifneq ($(wildcard $(HOSTRTL)/libxlsmp.so),)
		TEST_FOLDERS += $(wildcard tlomp-*)
  endif
  ifeq ($(machine),ppc64)
		TEST_FOLDERS += $(wildcard tp7-*)
  endif
  ifeq ($(machine),ppc64le)
		TEST_FOLDERS += $(wildcard tp8-*)
  endif
  ifeq ($(machine),x86_64)
		TEST_FOLDERS += $(wildcard tx86-*)
  endif

endif

TEST_FOLDERS_BLD_STATIC := $(addsuffix _static_bld,$(TEST_FOLDERS))
TEST_FOLDERS_RUN_STATIC := $(addsuffix _static_run,$(TEST_FOLDERS))
TEST_FOLDERS_BLD_DYNAMIC := $(addsuffix _dyn_bld,$(TEST_FOLDERS))
TEST_FOLDERS_RUN_DYNAMIC := $(addsuffix _dyn_run,$(TEST_FOLDERS))

#
# rule to run all the tests in static format
#

run_all_static: bld_all_static
	@rm -rf .omptests_failed
	@for i in $(TEST_FOLDERS) ; do \
	  $(MAKE) -C $$i run ; if [ $$? -ne 0 ] ; then touch .omptests_failed ; fi ; true; \
	done
	@if [ -f '.omptests_failed' ] ; \
		then echo " ---> Tests failed 8^( !!!" ; false ; fi
	@(printenv OMP_TARGET_OFFLOAD | grep "DISABLED" >/dev/null && echo " ---> Test ran on the host") || echo  " ---> Test ran on the device"
	@echo " ---> All tests completed successfully!!!"

runonly_all_static:
	@for i in $(TEST_FOLDERS) ; do \
	  echo ; \
	  echo "EXECUTION  TEST: $$i" ; \
	  $(MAKE) -C $$i run ; \
	  echo "EXECUTION  TEST: $$i;    STATUS:$$?" ; \
	done
	@(printenv OMP_TARGET_OFFLOAD | grep "DISABLED" >/dev/null && echo " ---> Test ran on the host") || echo  " ---> Test ran on the device"
	@echo " ---> All tests completed successfully!!!"

bld_all_static: $(TEST_FOLDERS_BLD_STATIC)
	@echo " ---> All tests built successfully!!!"

%_static_bld:
	@echo ; \
	echo "COMPILATION  TEST: $@" ; \
	$(MAKE) -C $(@:_static_bld=) static ; \
	echo "COMPILATION  TEST: $@;    STATUS:$$?" ;

mp_error: $(TEST_FOLDERS_RUN_STATIC)

%_static_run:
	@$(MAKE) -C $(@:_static_run=) run

#
# rule to run all the tests in shared lib format
#

run_all_dynamic: bld_all_dynamic
	@$(MAKE) -C fast_test static
	@rm -rf fast_test/list
	@for i in $(TEST_FOLDERS) ; do \
	  echo $(shell pwd)/$$i >> fast_test/list; \
	done
	@$(MAKE) -C fast_test run
	@(printenv OMP_TARGET_OFFLOAD | grep "DISABLED" >/dev/null && echo " ---> Test ran on the host") || echo  " ---> Test ran on the device"
	@echo " ---> All tests completed successfully!!!"

bld_all_dynamic: $(TEST_FOLDERS_BLD_DYNAMIC)
	@echo " ---> All tests built successfully!!!"

%_dyn_bld:
	$(MAKE) -C $(@:_dyn_bld=) dynamic

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
