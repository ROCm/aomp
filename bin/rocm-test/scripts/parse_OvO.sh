#! /bin/bash

# Parse log file from running OvO tests to gather
# passing and failing tests.
#
# Expected work flow:
#  ./run_ovo.sh log <optional: /prefix/for/output>
#  ./parse_OvO.sh <optional: /prefix/for/output>

realpath=`realpath $0`
aomp_regex='(.*\/aomp)\/'
[[ "$realpath" =~ $aomp_regex ]]
aompdir=${BASH_REMATCH[1]}

testname_regex='^([^>].*)'
group_regex='>> .*\/(.*\/.*\/.*)$'
compile_regex='>>> compilation error'
runtime_regex='>>> runtime error'
wrong_value_regex='>>> wrong value'
success_regex='>>> success'

cd "$HOME/git/aomp-test/OvO"
infile=`ls | grep "ovo.run.log"`

# Clean up before parsing
if [ -e passing-tests.txt ]; then
  rm passing-tests.txt
fi
if [ -e failing-tests.txt ]; then
  rm failing-tests.txt
fi
if [ -e make-fail.txt ]; then
  rm make-fail.txt
fi

function parse(){
  while read -r line; do
    local ovotest=""
    # Get test group
    if [[ "$line" =~ $group_regex ]]; then
      ovogroup=${BASH_REMATCH[1]}
      error=""
    # Get compile fail
    elif [[ "$line" =~ $compile_regex ]]; then
      error="Compilation Error"
    # Get runtime error
    elif [[ "$line" =~ $runtime_regex ]]; then
      error="Runtime Error"
    # Get runtime wrong value
    elif [[ "$line" =~ $wrong_value_regex ]]; then
      error="Wrong Value"
    # Get successful test
    elif [[ "$line" =~ $success_regex ]]; then
      error=""
    # Get test name
    elif [[ "$line" =~ $testname_regex ]]; then
      ovotestname=${BASH_REMATCH[1]}
        if [ "$error" != "" ]; then
          if [ "$error" == "Compilation Error" ]; then
            filename=make-fail.txt
          else
            filename=failing-tests.txt
          fi
        else
          filename=passing-tests.txt
        fi
      echo "$ovogroup/$ovotestname" >> $filename
    fi
  done < "$infile"
}

parse
