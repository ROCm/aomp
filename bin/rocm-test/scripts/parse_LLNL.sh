#! /bin/bash

# Parse log file from running LLNL tests to gather
# passing and failing tests.
#
# Expected work flow:
#  ./check_LLNL.sh log <optional: /prefix/for/output>
#  ./parse_LLNL.sh <optional: /prefix/for/output>

realpath=`realpath $0`
aomp_regex='(.*\/aomp)\/'
[[ "$realpath" =~ $aomp_regex ]]
aompdir=${BASH_REMATCH[1]}

testname_regex='.*(test_\S*)'
compile_regex='Compilation.*failed'
runtime_regex='Running.+\.\.\.\s+([A-Z]*[a-z]*)'

cd "$aompdir/test/LLNL/openmp5.0-tests" || exit 1
declare -a infiles
infiles=( LLNL.run.log* )
if [ "${#infiles[@]}" -ne 1 ]; then
  echo "Expected to find a single result file, bailing out" >&2
  exit 1
fi
infile=${infiles[0]}

# Clean up before parsing
if [ -e passing-tests.txt ]; then
  rm passing-tests.txt
fi
if [ -e failing-tests.txt ]; then
  rm failing-tests.txt
fi
if [ -e xfail-tests.txt ]; then
  rm xfail-tests.txt
fi
if [ -e make-fail.txt ]; then
  rm make-fail.txt
fi

function parse(){
  while read -r line; do
    local llnltest=""
    # Get compile fails
    if [[ "$line" =~ $compile_regex ]]; then
      [[ "$line" =~ $testname_regex ]]
      llnltest=${BASH_REMATCH[1]}
      echo "$llnltest" >> make-fail.txt
    fi
    # Get runtime results
    if [[ "$line" =~ $runtime_regex ]]; then
      result=${BASH_REMATCH[1]}
      if [ "$result" == "Passed" ]; then
        outfile="passing-tests.txt"
      else
        outfile="failing-tests.txt"
      fi
      [[ "$line" =~ $testname_regex ]]
      llnltest=${BASH_REMATCH[1]}
      if [ "$outfile" = "failing-tests.txt" ] && \
         grep -Fq "$llnltest" xfail-list; then
        outfile="xfail-tests.txt"
      fi
      echo "$llnltest" >> $outfile
    fi
  done < "$infile"
}

parse
