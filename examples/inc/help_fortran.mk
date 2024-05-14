#-----------------------------------------------------------------------
#
#  help_fortran.mk : This the help text for Make help in Fortran examples
#

help:
	@echo
	@echo "Source:			$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Compiler install dir:	$(LLVM_INSTALL_DIR)"
	@echo "Compiler name:		$(LLVM_COMPILER_NAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Target triple:		$(LLVM_GPU_TRIPLE)"
	@echo "Compiler: 		$(FORT)"
	@echo "Compile flags:		$(FFLAGS)"
	@echo "Linkflags:		$(LFLAGS)"
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
	@echo "  LLVM_GPU_ARCH        Target GPU, e.g. sm_30, gfx90a"
	@echo

