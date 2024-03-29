SHELL=/bin/sh
BENCHMARK=mg
BENCHMARKU=MG
DEFINES=-DCRPL_COMP=0
PXM=omp_offloading

include ../config/make.def

PROGRAM  = $(BINDIR)/$(BENCHMARK).$(CLASS).x
CCOMPILE = $(CC)  -c $(C_INC) $(CFLAGS)
CCOMPILE_pp = $(CC_pp)  -c $(C_INC_pp) $(CFLAGS_pp)

# Class "U" is used internally by the setparams program to mean
# "unknown". This means that if you don't specify CLASS=
# on the command line, you'll get an error. It would be nice
# to be able to avoid this, but we'd have to get information
# from the setparams back to the make program, which isn't easy. 
CLASS=U

default:: ${PROGRAM}

# This makes sure the configuration utility setparams 
# is up to date. 
# Note that this must be run every time, which is why the
# target does not exist and is not created. 
# If you create a file called "config" you will break things. 
config:
	@cd ../sys; ${MAKE} all
	../sys/setparams ${BENCHMARK} ${CLASS}

CONFIG_PATH=../config
COMMON=../common
${COMMON}/${RAND}.o: ${COMMON}/${RAND}.c ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} ${RAND}.c

${COMMON}/print_results.o: ${COMMON}/print_results.c ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} print_results.c

${COMMON}/c_print_results.o: ${COMMON}/c_print_results.c ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} c_print_results.c

${COMMON}/timers.o: ${COMMON}/timers.c ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} timers.c

${COMMON}/c_timers.o: ${COMMON}/c_timers.c ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} c_timers.c

${COMMON}/wtime.o: ${COMMON}/${WTIME} ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} ${MACHINE} -o wtime.o ${COMMON}/${WTIME}
# For most machines or CRAY or IBM
#	cd ${COMMON}; ${CCOMPILE} ${MACHINE} ${COMMON}/wtime.c
# For a precise timer on an SGI Power Challenge, try:
#	cd ${COMMON}; ${CCOMPILE} -o wtime.o ${COMMON}/wtime_sgi64.c

${COMMON}/c_wtime.o: ${COMMON}/${WTIME} ${CONFIG_PATH}/make.def
	cd ${COMMON}; ${CCOMPILE} -o c_wtime.o ${COMMON}/${WTIME}


# Normally setparams updates npbparams.h only if the settings (CLASS)
# have changed. However, we also want to update if the compile options
# may have changed (set in ${CONFIG_PATH}/make.def). 
npbparams.h: ${CONFIG_PATH}/make.def
	@ echo make.def modified. Rebuilding npbparams.h just in case
	rm -f npbparams.h
	../sys/setparams ${BENCHMARK} ${CLASS}

# So that "make benchmark-name" works
${BENCHMARK}:  default
${BENCHMARKU}: default


OBJS = mg.o \
       ../common/print_results.o  \
       ../common/c_timers.o \
       ../common/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} 
.c.o:
	${CCOMPILE} ${DEFINES} $<

mg.o:		mg.c globals.h npbparams.h ../common/print_results.o ../common/c_timers.o ../common/wtime.o 


clean:
	- rm -f *.o *~ ../common/*.o *.w2c.c *.w2c.h *.i *.B *.t *.w2c.cu *.w2c.ptx *.spin *.s *.x
	- rm -f npbparams.h core



