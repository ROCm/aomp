#!/bin/bash
#
#  check_omptests.sh
#  Assumes run_omptests.sh has been executed.
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

pushd $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME
rm -f runtime-fails.txt
rm -f compile-fails.txt
rm -f passing-tests.txt

# Get Results
compile_fails=0
runtime_fails=0

# Count tests that start with t- or test-
total_tests=$(ls | grep "\(^t\-*\|^test\-\)" | wc -l)

# Count compile/runtime fails and successful tests
for directory in ./t-*/; do
  pushd $directory > /dev/null
  testname=`basename $(pwd)`
  diff results/stdout expected > /dev/null
  return_code=$?
  if [ $return_code != 0 ] && [ -e results/a.out ]; then
    reason=`grep -E 'Killed' results/stderr`
    echo $testname $reason >> $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME/runtime-fails.txt
  elif ! [[ -e results/a.out ]]; then
    echo $testname >> $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME/compile-fails.txt
  else
    if [ -e results/a.out ]; then
      echo $testname >> $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME/passing-tests.txt
    fi
  fi
  popd > /dev/null
done

# Add skip_list tests to runtime fails
sed -e 's/$/ Skipped/' skipped-tests.txt >> runtime-fails.txt

# Count compile failures
echo
echo -----Compile Fails-----
if [ -e compile-fails.txt ]; then
  cat compile-fails.txt
  compile_fails=$(wc -l < compile-fails.txt)
fi

# Count runtime failures
# Add tests that were skipped to avoid soft hang
echo
echo -----Runtime Fails-----
runtime_fails=$(ls | grep "^test\-" | wc -l)
if [ -e runtime-fails.txt ]; then
  echo
  cat runtime-fails.txt
  ((runtime_fails=$(wc -l < runtime-fails.txt)))
fi

echo
echo -----Passing Tests-----
if [ -e passing-tests.txt ]; then
  cat passing-tests.txt
  passing_tests=$(wc -l < passing-tests.txt)
else
  passing_tests=0
fi

# Get final pass rate
if [ "$passing_tests" == "$total_tests" ]; then
  pass_rate=100
else
  # The calculation results in extra zeros that can be removed with sed
  pass_rate=`bc -l <<< "scale=4; ($passing_tests/$total_tests) * 100" | sed -E "s/([0-9]+\.[0-9]+)00/\1/g"`
fi

echo
echo ----- Results -----
echo Compile Fails: $compile_fails
echo Runtime Fails: $runtime_fails

echo Successful Tests: $passing_tests/$total_tests
echo Pass Rate: $pass_rate%
echo -------------------
echo

# Log Results
{
  echo
  echo ----- Results -----
  echo Compile Fails: $compile_fails
  echo Runtime Fails: $runtime_fails

  echo Successful Tests: $passing_tests/$total_tests
  echo Pass Rate: $pass_rate%
  echo -------------------
  echo
} >> omptests_run_$log
popd
