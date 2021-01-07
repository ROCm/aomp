#!/bin/bash
#
# Runs all check scripts in examples
#

#Text Colors
RED="\033[0;31m"
GRN="\033[0;32m"
BLU="\033[0;34m"
ORG="\033[0;33m"
BLK="\033[0m"

script_dir=$(dirname "$0")
pushd $script_dir
path=$(pwd)

echo ""
echo -e "$ORG"RUNNING ALL TESTS IN: $path"$BLK"
echo ""

echo "************************************************************************************" 
echo "                   A non-zero exit code means a failure occured." 
echo "************************************************************************************"

#Loop over all directories and run the check script
cloc=""
if [ "$EPSDB" != "1" ]; then
  LIST="fortran hip openmp cloc"
else
  LIST="fortran openmp"
fi

for directory in ./$LIST/; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
    script=check_$base.sh
    ./$script
    echo ""
    )
done
echo -e "$ORG"FINAL RESULTS:"$BLK"
for directory in $LIST ; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
    cat check-$base.txt
  )
done
popd
