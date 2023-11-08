#!/bin/bash

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

echo " >>> Configure ..."
cmake -B build -S .                                                            \
-DAOMP_DIR=${AOMP_DIR}                                                         \
-DTGT_OFFLOAD_ARCH=${TGT_OFFLOAD_ARCH}

echo " >>> Clean & Build ..."
cmake --build build --clean-first --parallel || exit 1

echo " >>> Run ..."
./build/$PROJECT_NAME || exit 1

echo " >>> DONE!"