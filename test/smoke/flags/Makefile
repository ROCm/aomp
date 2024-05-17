include ../../Makefile.defs

TESTNAME     = flags
TESTSRC_MAIN = flags.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

file="options.txt"
TESTNAMES_ALL = $(basename $(TESTSRC_ALL))

all: $(TESTNAME)

.PRECIOUS: %.o %.ll %.ll.s %.s %.s.o

$(TESTNAME): $(TESTSRC_ALL)
	@./run_options.sh $(AOMP_GPU)


compile:
ifdef nvidia_targets
	$(CC) $(make_options) $(nvidia_targets) $(march) $(cuda) -o $(TESTNAME)
else ifdef march
	$(SETENV) $(CC) $(make_options) $(march) -o $(TESTNAME)
else ifdef make_options
	$(CC) $(make_options) -o $(TESTNAME)
else
	$(SETENV) $(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $(LINK_FLAGS) $(wordlist 2,$(words $^),$^) -o $(TESTNAME)
endif

run:
	@./run_options.sh run $(AOMP_GPU) 2>&1 | tee $@.log

check: compile
	@path=`pwd`; \
	base=`basename $$path`; \
	( \
	flock -e 9 && echo  "" >> ../check-smoke.txt; \
	if   (./$(TESTNAME)); then \
		echo $$base $$test_num return code: $$? >> ../check-smoke.txt; \
		echo "" >> ../check-smoke.txt; \
		echo $$base $$test_num  >> ../passing-tests.txt; \
	else  echo $$base $$test_num return code: $$? >> ../check-smoke.txt; \
		echo "" >> ../check-smoke.txt; \
		echo $$base $$test_num  >> ../failing-tests.txt; \
	fi; \
	)9>../lockfile;

clean::
	rm -f $(TESTNAME) $(TESTNAME).a llbin sbin obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.log
