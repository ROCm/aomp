#!/bin/bash
#
# Runs each command in the options.txt file and returns error code.
# The Makefile will run this script twice when using 'make run'
# First run is to compile, second is to run for output in run.log.

# Input file
file="options.txt"
# Regex to search for "march=gfxXXX or sm_XX"
# run_options will set march to be either AOMP_GPU or auto-dect with mygpu utility
march_regex="(march=AOMP_GPU_or_auto_detect)"
# Regex to search for OFFLOAD_DEBUG
debug_regex="(OFFLOAD_DEBUG=([0-9]))"
# Regex to search for Nvidia cards
target_regex="(-fopenmp-[a-z]*=[a-z,-]*).*(-Xopenmp-[a-z]*=[a-z,-]*)"

UNAMEP=`uname -m`
if [[ $UNAMEP == "ppc64le" ]] ; then
   AOMP_CPUTARGET=-target ppc64le-linux-gnu
fi

path=$(pwd) && base=$(basename $path)
# Read file, replace march with correct GPU, add Nvidia options if necessary and keep track of execution number
test_num=0;
while read -r line; do
  ((test_num++))
  # GPU is involved
  if [[ "$line" =~ $march_regex ]]; then
    march_match=${BASH_REMATCH[1]}
    # Remove march from command and replace with correct version
    line=${line/"-$march_match"}
    mygpu=$AOMP_GPU
    if [ -z $mygpu ]; then
      echo -e "$RED"AOMP_GPU NOT SET, USING MYGPU UTILITY!"$BLK"
      if [[ $1 != "run" ]]; then
        mygpu="$1"
      fi
      if [[ $1 == "run" ]]; then
        mygpu="$2"
      fi
    fi

    # Check for OFFLOAD_DEBUG
    if [[ "$line" =~ $debug_regex ]]; then
      debug_match=${BASH_REMATCH[1]}
      line=${line/"$debug_match"}
      export $debug_match
    fi

    # NVIDIA system, add nvptx targets, cuda, and remove amdgcn targets. This is done for testing purpose to avoid rewriting original amd command.
    if [[ "$mygpu" == *"sm"* ]]; then
      if [[ "$line" =~ $target_regex ]]; then
        target_match="${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        line=${line/"$target_match"}
        nvidia_args="-fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda"
        cuda_args="-L/usr/local/cuda/targets/${UNAMEP}-linux/lib -lcudart"
      fi
    fi
    # GPU compilation or run, send variables to make
    if [[ $1 != "run" ]]; then
      make --no-print-directory make_options="$line" nvidia_targets="$nvidia_args" march="-march=$mygpu" cuda="$cuda_args" compile
    fi
    if [[ $1 == "run" ]]; then
      make --no-print-directory make_options="$line" nvidia_targets="$nvidia_args" march="-march=$mygpu" cuda="$cuda_args" test_num=$test_num check
    fi
    if [ $? -ne 0 ]; then
      echo "$base $test_num: Make Failed" >> ../make-fail.txt
    fi
  else # Host compilation or run, GPU not detected on input line, no need to pass other variables to make
    if [[ $1 != "run" ]]; then
      make --no-print-directory make_options="$line" compile
    fi
    if [[ $1 == "run" ]]; then
      make --no-print-directory make_options="$line" test_num=$test_num check
    fi
  fi
  # Host not successfull
  if [ $? -ne 0 ]; then
    echo "$base $test_num: Make Failed" >> ../make-fail.txt
  else
    rm flags
  fi
  echo ""

  # Reset OFFLOAD_DEBUG
  reset_debug=$OFFLOAD_DEBUG
  if [ -z $reset_Debug ]; then
    export OFFLOAD_DEBUG=0
  fi

done < "$file"
