#!/bin/bash

usage() { echo "Usage: $0 [-a <AOMP directory>]"                               \
                        " [-c <Compiler path>]"                                \
                        " [-f <Compiler flags>]"                               \
                        " [-l <Switch for 'LLVM IR' demo mode>]"               \
                        " [-m <Switch for 'AMDGPU assembly' demo mode>]"       \
                        " [-t <Test source file>]"                             \
          1>&2; exit 1; }

while getopts "lm :a:c:f:t:" o; do
    case "${o}" in
        a)
            AOMP=${OPTARG}
            ;;
        c)
            CC=${OPTARG}
            ;;
        f)
            CFLAGS=${OPTARG}
            ;;
        l)
            LLVMIR=1
            ;;
        m)
            ASSEMBLY=1
            ;;
        t)
            TESTSRC=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

# Set 'PROJECT_NAME' to the parent directory's name
PROJECT_NAME=$(basename $(realpath $(pwd)))

# If 'AOMP' was not specified, fallback to user's AOMP directory
if [[ -z ${AOMP} ]]; then
  AOMP="/home/$USER/rocm/aomp"
fi

# Check if AOMP directory exists
if [ ! -d ${AOMP} ]; then
  echo "ERROR: AOMP directory '${AOMP}' does not exist!"
  exit 2
fi

# If 'TESTSRC' was not specified: exit
if [[ -z ${TESTSRC} ]]; then
  echo "ERROR: No source file specified! Use: [-t <Test source file>]"
  exit 2
fi

# If 'CC' was not specified: use user's clang
if [[ -z ${CC} ]]; then
  CC=${AOMP}/bin/clang
fi

# Atomic optimization kinds
OPT_DPP="  -mllvm --amdgpu-atomic-optimizer-strategy=DPP"
OPT_ITER=" -mllvm --amdgpu-atomic-optimizer-strategy=Iterative"
OPT_NONE=" -mllvm --amdgpu-atomic-optimizer-strategy=None"

GEN_ASM_DPP="$CC -c -S $CFLAGS $OPT_DPP $TESTSRC"
GEN_ASM_ITER="$CC -c -S $CFLAGS $OPT_ITER $TESTSRC"
GEN_ASM_NONE="$CC -c -S $CFLAGS $OPT_NONE $TESTSRC"

# Extract the embedded offload object from the given assembly input and format
# the embedded object such that it becomes amenable for filechecking.
# FixMe: GET_EMBEDDED currently does nothing, but since we check for AMDGPU ASM
#        the lit-test will still work, since the other instructions are X86 ASM
#        The idea is / was to only slice embedded objects from the given file.
GET_EMBEDDED="/.Lllvm.embedded.object:/,/.size/p"
SUB_NEWLINE="s/\\\\n/\\n/g"
SUB_TABULATOR="s/\\\\t/\\t/g"
EXTRACT_EMBEDDED=" sed -e $GET_EMBEDDED -z -e $SUB_NEWLINE -e $SUB_TABULATOR"

COMPILE_DPP="$CC $CFLAGS $OPT_DPP $TESTSRC"
COMPILE_ITER="$CC $CFLAGS $OPT_ITER $TESTSRC"
COMPILE_NONE="$CC $CFLAGS $OPT_NONE $TESTSRC"

# Usual RUN or CHECK recipe handling
if [[ -z ${LLVMIR} && -z ${ASSEMBLY} ]]; then
  # Generate AMDGPU assembly files, format output and then filecheck for the
  # expected instructions within the embedded object.
  echo " > FileChecks"
  ${GEN_ASM_DPP}  -o - | ${EXTRACT_EMBEDDED} | ${AOMP}/bin/FileCheck $TESTSRC  \
    --check-prefix=DPP || exit 1
  ${GEN_ASM_ITER} -o - | ${EXTRACT_EMBEDDED} | ${AOMP}/bin/FileCheck $TESTSRC  \
    --check-prefix=ITERATIVE || exit 1
  ${GEN_ASM_NONE} -o - | ${EXTRACT_EMBEDDED} | ${AOMP}/bin/FileCheck $TESTSRC  \
    --check-prefix=NONE || exit 1

  # Compile and execute. Performing a simple check for an expected result.
  echo " > Compile & execute"
  ${COMPILE_DPP}  -o ${PROJECT_NAME}_dpp  && ./${PROJECT_NAME}_dpp  || exit 1
  ${COMPILE_ITER} -o ${PROJECT_NAME}_iter && ./${PROJECT_NAME}_iter || exit 1
  ${COMPILE_NONE} -o ${PROJECT_NAME}_none && ./${PROJECT_NAME}_none || exit 1
fi

# LLVMIR recipe handling
if [[ ${LLVMIR} == 1 ]]; then
  echo " > LLVMIR"
  ${GEN_ASM_DPP}  -emit-llvm -o  ${PROJECT_NAME}_dpp.ll
  sed -i "1s/^/; ${OPT_DPP}\n/"  ${PROJECT_NAME}_dpp.ll
  ${GEN_ASM_ITER} -emit-llvm -o  ${PROJECT_NAME}_iter.ll
  sed -i "1s/^/; ${OPT_ITER}\n/" ${PROJECT_NAME}_iter.ll
  ${GEN_ASM_NONE} -emit-llvm -o  ${PROJECT_NAME}_none.ll
  sed -i "1s/^/; ${OPT_NONE}\n/" ${PROJECT_NAME}_none.ll
fi

# ASSEMBLY recipe handling
if [[ ${ASSEMBLY} == 1 ]]; then
  echo " > ASSEMBLY"
  ${GEN_ASM_DPP}  -o - | ${EXTRACT_EMBEDDED} > ${PROJECT_NAME}_dpp.s
  sed -i "1s/^/; ${OPT_DPP}\n/" ${PROJECT_NAME}_dpp.s
  ${GEN_ASM_ITER} -o - | ${EXTRACT_EMBEDDED} > ${PROJECT_NAME}_iter.s
  sed -i "1s/^/; ${OPT_ITER}\n/" ${PROJECT_NAME}_iter.s
  ${GEN_ASM_NONE} -o - | ${EXTRACT_EMBEDDED} > ${PROJECT_NAME}_none.s
  sed -i "1s/^/; ${OPT_NONE}\n/" ${PROJECT_NAME}_none.s
fi

echo " > OK"
