#-----------------------------------------------------------------------
#
#  obin.mk
#
#  ----   Demo compile and link in two steps, object is saved.
#         Only for those examples that have a single source file. 
#         Note: ensure TESTNAME has no blanks
#
$(TESTNAME).o: $(TESTSRC)
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@

obin: $(TESTNAME).o
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run_obin: obin
	$(RUNENV) ./obin

