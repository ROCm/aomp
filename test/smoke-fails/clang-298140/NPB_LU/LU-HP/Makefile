SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=LU
DEFINES=-DCRPL_COMP=0
PXM=omp_offloading
VEC=

include ../config/make.def

include ../sys/make.common

OBJS = lu.o read_input.o \
       domain.o setcoeff.o setbv.o exact.o setiv.o \
       erhs.o ssor$(VEC).o rhs$(VEC).o l2norm.o \
       jacld.o blts$(VEC).o jacu.o buts$(VEC).o error.o \
       calcnp.o pintgr.o verify.o ${COMMON}/print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/wtime.o


# npbparams.h is included by applu.incl
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# applu.incl: npbparams.h

${PROGRAM}: config
	@if [ x$(VERSION) = xvec ] ; then	\
		${MAKE} VEC=_vec exec;		\
	elif [ x$(VERSION) = xVEC ] ; then	\
		${MAKE} VEC=_vec exec;		\
	else					\
		${MAKE} exec;			\
	fi

exec: $(OBJS)
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} ${DEFINES} $<

lu.o:		lu.c applu.incl npbparams.h
calcnp.o:   calcnp.c applu.incl npbparams.h
blts$(VEC).o:	blts$(VEC).c
buts$(VEC).o:	buts$(VEC).c	
erhs.o:		erhs.c applu.incl npbparams.h
error.o:	error.c applu.incl npbparams.h
exact.o:	exact.c applu.incl npbparams.h
jacld.o:	jacld.c applu.incl npbparams.h
jacu.o:		jacu.c applu.incl npbparams.h
l2norm.o:	l2norm.c
pintgr.o:	pintgr.c applu.incl npbparams.h
read_input.o:	read_input.c applu.incl npbparams.h
rhs$(VEC).o:	rhs$(VEC).c applu.incl npbparams.h
setbv.o:	setbv.c applu.incl npbparams.h
setiv.o:	setiv.c applu.incl npbparams.h
setcoeff.o:	setcoeff.c applu.incl npbparams.h
ssor$(VEC).o:	ssor$(VEC).c applu.incl npbparams.h
domain.o:	domain.c applu.incl npbparams.h
verify.o:	verify.c applu.incl npbparams.h

clean:
	- /bin/rm -f npbparams.h
	- /bin/rm -f *.o *~ ../common/*.o *.w2c.cu *.w2c.h *.w2c.c *.i *.B *.spin *.t *.s *.gpu *.x *.ptx *.n00*.h 
