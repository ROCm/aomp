#!/bin/bash
#===--------- libm/libm.sh -----------------------------------------------===
#
#                The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===

GPUS=${GPUS:-"gfx700 gfx701 gfx801 gfx803 gfx900 "}
GPUS+="sm_30 sm_35 sm_50 sm_60 "
for i in $GPUS ; do
  echo
  echo Building DBCL for $i
  echo
  AOMP_GPU=$i make
done
