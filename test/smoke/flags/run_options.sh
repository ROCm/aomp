#!/bin/bash
#
# Runs each command in the optionst.txt file and returns error code.
#

file="options.txt"
#Regex to search for "march=gfxXXX or sm_XX"
march_regex="(march=[a-z]+[0-9]*_?[0-9]+)"
path=$(pwd) && base=$(basename $path)
#Read file and replace march with correct GPU and keep track of execution number
test_num=0;
while read -r line; do
  ((test_num++))
  #if GPU is involved
  if [[ "$line" =~ $march_regex ]]; then
    march_match=${BASH_REMATCH[1]}
    #remove march from command and replace with correct version
    temp_line=${line/"-$march_match"}
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

    #If NVIDIA system, add nvptx targets, cuda, and remove amdgcn targets. This is done for testing purpose to avoid rewriting original amd command.
    if [[ "$AOMP_GPU" == *"sm"* ]]; then
      target_regex="(-fopenmp-[a-z]*=[a-z,-]*).*(-Xopenmp-[a-z]*=[a-z,-]*)"
      if [[ "$line" =~ $target_regex ]]; then
        target_match="${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        temp_line=${temp_line/"$target_match"}
        nvidia_args="-fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda"
        cuda_args="-L/usr/local/cuda/targets/x86_64-linux/lib -lcudart"
      fi
    fi
    #send variables to make
    if [[ $1 != "run" ]]; then
      make --no-print-directory make_options="$temp_line" nvidia_targets="$nvidia_args" march="-march=$mygpu" cuda="$cuda_args" compile
    fi
    if [[ $1 == "run" ]]; then
      make --no-print-directory make_options="$temp_line" nvidia_targets="$nvidia_args" march="-march=$mygpu" cuda="$cuda_args" test_num=$test_num check
    fi
    if [ $? -ne 0 ]; then
      echo "$base $test_num: Make Failed" >> ../make-fail.txt
    fi
    else
      if [[ $1 != "run" ]]; then
        make --no-print-directory make_options="$line" compile
      fi
      if [[ $1 == "run" ]]; then
        make --no-print-directory make_options="$line" test_num=$test_num check
      fi
    fi
    if [ $? -ne 0 ]; then
      echo "$base $test_num: Make Failed" >> ../make-fail.txt
    else
      rm flags
  fi
  echo ""
done < "$file"
