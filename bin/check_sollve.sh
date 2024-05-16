#!/bin/bash
#
#  check_sollve.sh
#  Assumes run_sollve.sh has been executed.
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

set -e

pushd $AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME
testname_regex='Test\sname":\s"(.*)"'
compiler_regex='Compiler\sresult":\s"([A-Z]+)'
runtime_regex='Runtime\sresult":\s"([A-Z]+)'
function parse(){
  local file=$2
  while read -r line; do
    local sollvetest=""
    if [[ "$line" =~ $testname_regex ]]; then
      sollvetest=${BASH_REMATCH[1]}
    elif [[ "$line" =~ $compiler_regex ]]; then
      compresult=${BASH_REMATCH[1]}
    elif [[ "$line" =~ $runtime_regex ]]; then
      runresult=${BASH_REMATCH[1]}
    fi
    if [ "$sollvetest" != "" ];then
      results[$sollvetest]="$compresult"
      results[$sollvetest]+="$runresult"
      compresult=""
      runresult=""
    fi
  done < "$file"
}

function checkstatus(){
  tmppassfile="results_report$1/tmppassing-tests.txt"
  tmpcompfailfile="results_report$1/tmpmake-fail.txt"
  tmprunfailfile="results_report$1/tmpfailing-tests.txt"

  local val=$2
  if [ "$val" == "PASSPASS" ]; then
    echo $i >> $tmppassfile
  elif [ "$val" == "FAIL" ]; then
    echo $i >> $tmpcompfailfile
  elif [ "$val" == "PASSFAIL" ]; then
    echo $i >> $tmprunfailfile
  fi

}

function checkresult(){
  passfile="results_report$1/passing-tests.txt"
  compfailfile="results_report$1/make-fail.txt"
  runfailfile="results_report$1/failing-tests.txt"

  rm -f $tmppassfile $tmpcompfailfile $tmprunfailfile

  for i in "${!results[@]}"; do
    local val=${results[$i]}
    checkstatus $1 $val
  done

  if [ -e "$tmppassfile" ]; then
    sort $tmppassfile > results_report$1/passing-tests.txt
  fi
  if [ -e "$tmpcompfailfile" ]; then
    sort $tmpcompfailfile > results_report$1/make-fail.txt
  fi
  if [ -e "$tmprunfailfile" ]; then
    sort $tmprunfailfile > results_report$1/failing-tests.txt
  fi
  rm -f $tmppassfile $tmpcompfailfile $tmprunfailfile
}

# Get all openmp versions in sollve from available reports
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
  base_dir=$AOMP_REPOS_TEST/$AOMP_SOLVV_REPO_NAME
  cd $base_dir

  if [ "$2" != "" ]; then
    prefix=$2
    log="$prefix/sollve.run.log.$date"
  else
    log="$base_dir/sollve.run.log.$date"
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
