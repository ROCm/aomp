#!/bin/bash
#
#  check_openmpvv.sh
#  Assumes run_openmpvv.sh has been executed.
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

set -e

pushd $AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME
testname_regex='Test\sname":\s"(.*)"'
compiler_regex='Compiler\sresult":\s"([A-Z]+)'
runtime_regex='Runtime\sresult":\s"([A-Z]+)'
function parse(){
  local file=$2
  while read -r line; do
    local openmpvvtest=""
    if [[ "$line" =~ $testname_regex ]]; then
      openmpvvtest=${BASH_REMATCH[1]}
    elif [[ "$line" =~ $compiler_regex ]]; then
      compresult=${BASH_REMATCH[1]}
    elif [[ "$line" =~ $runtime_regex ]]; then
      runresult=${BASH_REMATCH[1]}
    fi
    if [ "$openmpvvtest" != "" ];then
      results[$openmpvvtest]="$compresult"
      results[$openmpvvtest]+="$runresult"
      compresult=""
      runresult=""
    fi
  done < "$file"
}

# Evaluate a given test's status, then store the result in corresponding
# temporary files, grouped by OpenMP version.
# Status can be:
#   'PASSPASS' (Compile: PASS, Run: PASS)
#   'FAIL'     (Compile: FAIL, Run: consequently not available)
#   'PASSFAIL' (Compile: PASS, Run: FAIL)
function checkstatus(){
  local omp_version="$1"
  local testname="$2"
  local status="$3"
  tmppassfile="results_report${omp_version}/tmppassing-tests.txt"
  tmpcompfailfile="results_report${omp_version}/tmpmake-fail.txt"
  tmprunfailfile="results_report${omp_version}/tmpfailing-tests.txt"

  if [ ${status} == "PASSPASS" ]; then
    echo ${testname} >> ${tmppassfile}
  elif [ ${status} == "FAIL" ]; then
    echo ${testname} >> ${tmpcompfailfile}
  elif [ ${status} == "PASSFAIL" ]; then
    echo ${testname} >> ${tmprunfailfile}
  fi
}

# Evaluate the test suite's results, checking each test's status and then
# consolidating the results, grouped by OpenMP version and outcome.
# Here, OpenMP versions are strings consisting of two digits, e.g. '45' or '50'.
function checkresult(){
  local omp_version="$1"
  passfile="results_report${omp_version}/passing-tests.txt"
  compfailfile="results_report${omp_version}/make-fail.txt"
  runfailfile="results_report${omp_version}/failing-tests.txt"

  rm -f ${tmppassfile} ${tmpcompfailfile} ${tmprunfailfile}

  for testname in "${!results[@]}"; do
    local status=${results[${testname}]}
    checkstatus ${omp_version} ${testname} ${status}
  done

  if [ -e "${tmppassfile}" ]; then
    sort ${tmppassfile} > results_report${omp_version}/passing-tests.txt
  fi
  if [ -e "${tmpcompfailfile}" ]; then
    sort ${tmpcompfailfile} > results_report${omp_version}/make-fail.txt
  fi
  if [ -e "${tmprunfailfile}" ]; then
    sort ${tmprunfailfile} > results_report${omp_version}/failing-tests.txt
  fi
  rm -f ${tmppassfile} ${tmpcompfailfile} ${tmprunfailfile}
}

# Get all openmp versions in openmpvv from available reports
vers=`ls $s | grep results_report | grep -Eo [0-9][0-9]`

# Loop through results json for each openmp version and log pass/fails
for ver in $vers; do
  declare -A results
  parse $ver "results_report$ver/results.json"
  checkresult $ver
  unset results
done

# This log combines all pass/fails from various openmp versions into one file.
# Each test is prefixed with openmp version for clarity.
if [ "$1" == "log" ]; then
  date=${BLOG_DATE:-`date '+%Y-%m-%d'`}
  base_dir=$AOMP_REPOS_TEST/$AOMP_OPENMPVV_REPO_NAME
  cd $base_dir

  if [ "$2" != "" ]; then
    prefix=$2
    log="$prefix/openmpvv.run.log.$date"
  else
    log="$base_dir/openmpvv.run.log.$date"
  fi
  rm -f $parent_dir$log
  for ver in $vers; do
    cd $base_dir/results_report$ver
    files="make-fail.txt passing-tests.txt failing-tests.txt"
    for file in $files; do
      if [ -f $file ]; then
        # Add version in front of test for clarity
        sed -e "s/^/$ver-$file-/" $file > tmp$file
        cat tmp$file >> "$log"
        rm -f tmp$file
      fi
    done
  done
fi

exit
popd
