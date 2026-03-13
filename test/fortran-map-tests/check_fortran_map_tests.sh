#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier:  MIT

if [ $# -lt 1 ]; then
  echo "Error: You need to select execution mode."
  exit 1
fi

if [[ "$1" -eq 0 ]]; then
# These currently should all pass.
export OMP_TARGET_OFFLOAD=MANDATORY

echo "OFFLOAD MANDATORY"

./test.sh

elif [[ "$1" -eq 1 ]]; then
# Running these is more for experimental purposes, and future fixing.
# A lot will currently fail in host fallback mode.

export OMP_TARGET_OFFLOAD=DISABLED

echo "OFFLOAD DISABLED"

./test.sh

# Unset OMP_TARGET_OFFLOAD and remove files.

unset OMP_TARGET_OFFLOAD

fi

rm *.out
rm *.mod
