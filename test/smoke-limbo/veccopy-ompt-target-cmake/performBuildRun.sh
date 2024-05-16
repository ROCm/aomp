#!/bin/bash
set -x

usage() { echo "Usage: $0 [-a <AOMP directory>]" \
                        " [-t <Target offload architecture>]" \
          1>&2; exit 1; }

while getopts ":a:t:" o; do
    case "${o}" in
        a)
            AOMP_DIR=${OPTARG}
            ;;
        t)
            TGT_OFFLOAD_ARCH=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

# Set 'PROJECT_NAME' to the parent directory's name
PROJECT_NAME=$(basename $(pwd))

# Set 'build' directory for re-use
BUILD_DIR=build

# If 'AOMP_DIR' was not specified, fallback to user's AOMP directory
if [[ -z ${AOMP_DIR} ]]; then
  AOMP_DIR="/home/$USER/rocm/aomp"
fi

# If 'TGT_OFFLOAD_ARCH' was not specified, fallback to 'native'
if [[ -z ${TGT_OFFLOAD_ARCH} ]]; then
  TGT_OFFLOAD_ARCH="native"
fi

if [ ! -d ${AOMP_DIR} ]; then
  echo "WARNING: AOMP directory '${AOMP_DIR}' does not exist!"
fi

echo " >>> Clean ..."
git clean -fdx ./${BUILD_DIR}

echo " >>> Configure ..."
cmake -B ${BUILD_DIR} -S .                                                     \
-DAOMP_DIR=${AOMP_DIR}                                                         \
-DTGT_OFFLOAD_ARCH=${TGT_OFFLOAD_ARCH}

echo " >>> Build ..."
cmake --build ${BUILD_DIR} --clean-first --parallel || exit 1

echo " >>> Run ..."
./${BUILD_DIR}/${PROJECT_NAME} || exit 1

echo " >>> DONE!"