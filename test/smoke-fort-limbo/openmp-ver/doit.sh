#!/bin/bash

echo "Verifying initial test"
exp=202111                      # OpenMP 5.2 release date
echo "    Saw:"
grep "var1 = " test
echo "    Expected: $exp"
grep "var1 = $exp" test
mystat=$?
echo "    status: $mystat"
allstat=$(($allstat+$mystat))

echo "Checking all other flags"
declare -A flags=(
    [-fopenmp]=202111           # no specific version specified
    [-fopenmp-version=11]=199911
    [-fopenmp-version=20]=200011
    [-fopenmp-version=25]=200505
    [-fopenmp-version=30]=200805
    [-fopenmp-version=31]=201107
    [-fopenmp-version=40]=201307
    [-fopenmp-version=45]=201511
    [-fopenmp-version=50]=201811
    [-fopenmp-version=51]=202011
    [-fopenmp-version=52]=202111
)

IFS=$'\n' skeys=($(sort <<<"${!flags[@]}"))
unset IFS

for flag in ${skeys[@]}; do 
    cmd="${AOMP}/bin/${FLANG} -fopenmp $flag -cpp -E test.f90"
    exp=${flags[$flag]}
    echo $cmd
    echo "    Saw:"
    $cmd | grep "var1 = "
    echo "    Expected: $exp"
    $cmd | grep "var1 = $exp"
    mystat=$?
    echo "    status: $mystat"
    allstat=$(($allstat+$mystat))
done

echo "allstat: $allstat"
exit $allstat
