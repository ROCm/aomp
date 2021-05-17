#!/bin/bash

set -e

# Set up infra for running the examples in github.com/OpenMP under the aomp
# test environment. Borrows heavily from smoke.
#
# Specifically, downloads & checks out the above repo, then writes a
# directory structure containing makefiles very similar to those used
# by smoke. After running build.sh, the directory structure on success:
#
# - Examples # github repo, untouched by this script
# - C
#     - Makefile       # contains list of tests under C
#     - Makefile.defs  # symlink to smoke
#     - Makefile.rules # symlink to smoke
#     - Example_acquire_release.1
#       - Example_acquire_release.1.c # symlink into examples
#       - Makefile
#     - Further tests
# 
# - CXX # Two cases on main, but zero on v5.0.0 tag
# - F
# - F90
# make run in one of these directories will run the tests.

THISDIR=$PWD

REPONAME="Examples"

if [[ -d $REPONAME ]]
then
    echo "Found existing examples repo"
else
    echo "Cloning examples repo"
    git clone https://github.com/OpenMP/$REPONAME.git $REPONAME
    cd $REPONAME
    git checkout v5.0.0
    cd ..
fi


SDIR=$REPONAME/sources

FSRC=$(ls $SDIR/*.f)
CXXSRC=$(ls $SDIR/*.cpp)
F90SRC=$(ls $SDIR/*.f90)

function getname {
    BASE=$(basename $1)
    echo "${BASE%.*}"
}

function emitmakefile {
    MAKEFILENAME="$1"
    COMPILERVAR="$2"
    COMPILERINST="$3"
    NAME="$4"
    SUFFIX="$5"
    TARGETFIELD="$6"

    cat << EOF > $MAKEFILENAME
include ../../Makefile.defs

TESTNAME     = ${NAME}
TESTSRC_MAIN = ${NAME}${SUFFIX}
TESTSRC_AUX  =
TESTSRC_ALL  = \$(TESTSRC_MAIN) \$(TESTSRC_AUX)
${TARGETFIELD}
${COMPILERVAR}        = ${COMPILERINST} 
OMP_BIN      = \$(AOMP)/bin/\$(${COMPILERVAR})
CC           = \$(OMP_BIN) \$(VERBOSE)

include ../Makefile.rules

EOF

}

function emittestcases {
    SUFFIX="$1"
    DIRNAME="$2"
    COMPILERVAR="$3"
    COMPILERINST="$4"
    TARGETFIELD="$5"

    rm -rf $DIRNAME && mkdir -p $DIRNAME
    ln -s $THISDIR/../smoke/Makefile.defs $THISDIR/$DIRNAME/Makefile.defs 
    ln -s $THISDIR/../smoke/Makefile.rules $THISDIR/$DIRNAME/Makefile.rules 

    RUNNER=$THISDIR/$DIRNAME/Makefile
    
    cat <<'EOF'>$RUNNER
include Makefile.defs

TESTS_DIR = \
EOF

    
for S in $(ls $SDIR/*$SUFFIX); do
    NAME=$(getname $S)
    if grep -qP "@@linkable:[[:space:]]yes" $S; then
        mkdir -p $DIRNAME/$NAME
        ln -s $THISDIR/$S $THISDIR/$DIRNAME/$NAME/$NAME$SUFFIX

        echo "Creating test case $NAME$SUFFIX"
        echo "    $NAME \\" >> $RUNNER
        emitmakefile $THISDIR/$DIRNAME/$NAME/Makefile "$COMPILERVAR" "$COMPILERINST" "$NAME" "$SUFFIX" "$TARGETFIELD"
    else echo "Can't link $NAME$SUFFIX, skipping"
    fi
done


# Copied out of smoke
    cat <<'EOF'>>$RUNNER
#
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

EOF


}


FORTRANTARG='TARGET       = -fopenmp -O3 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU)'

emittestcases ".c" "C" "CLANG" "clang" ""

emittestcases ".cpp" "CXX" "CLANG" "clang++" ""

emittestcases ".f90" "F90" "FLANG" "flang" "$FORTRANTARG"

emittestcases ".f" "F" "FLANG" "flang" "$FORTRANTARG"



