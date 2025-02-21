#!/bin/bash
set -o pipefail
grep '^ XNACK' $1 | diff - ref.csv
exit $?
