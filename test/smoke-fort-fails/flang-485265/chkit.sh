#!/bin/bash
TEST_FILE=flang-485265-host-x86_64-unknown-linux-gnu-llvmir.mlir
if [ ! -f $TEST_FILE ]; then
  echo "File not found"
  exit 1
fi

NUMBER_OF_TBAA_TAGS=$( grep '^#tbaa_tag' $TEST_FILE | wc -l )

# We expect 4 alias analysis tags (one tag for every variable)
if [ $NUMBER_OF_TBAA_TAGS != "4" ]; then
  echo "Found " $NUMBER_OF_TBAA_TAGS " alias analysis tags"
  echo "Failure: Expected to find 4 different alias analysis tags"
  exit 1
fi

