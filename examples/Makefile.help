#-----------------------------------------------------------------------
#
#  Makefile.help : This the help text for Make help 
#

help:
	@echo
	@echo "Source:			$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Target triple:		$(LLVM_GPU_TRIPLE)"
	@echo "Compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make $(TESTNAME).o		// build object file "
	@echo " make obin		// Link object file to build binary "
	@echo " make run_obin		// Execute obin "
	@echo
	@echo " make clean"
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_INSTALL_DIR     LLVM installation directory"
	@echo "  LLVM_GPU_ARCH        Target GPU, e.g sm_30, gfx90a, etc"
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
	@echo

