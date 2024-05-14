#-----------------------------------------------------------------------
#
#  obin_fortran.mk
#
#  ----   Demo compile and link in two steps, object is saved.
#         Only for those examples that have a single source file. 
#         Note: ensure TESTNAME has no blanks
#
$(TESTNAME).o: $(TESTSRC)
	$(FORT) -c $(FFLAGS) $^ -o $@

obin: $(TESTNAME).o
	$(FORT) $(LFLAGS) $^ -o $@

run_obin: obin
	$(FORTENV) ./obin

