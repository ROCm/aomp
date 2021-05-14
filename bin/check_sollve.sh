#!/bin/bash
#
#  check_sollve.sh
#  Assumes run_sollve.sh has been executed.
#
set -e
pushd $HOME/git/aomp-test/sollve_vv
testname_regex='Test\sname":\s"(.*)"'
compiler_regex='Compiler\sresult":\s"([A-Z]+)'
runtime_regex='Runtime\sresult":\s"([A-Z]+)'
function parse(){
  local -n myarray=$1
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
      myarray[$sollvetest]="$compresult"
      myarray[$sollvetest]+="$runresult"
      compresult=""
      runresult=""
    fi
  done < "$file"
}

function checkresult(){
  tmppassfile="results_report$2/tmppassing-tests.txt"
  tmpcompfailfile="results_report$2/tmpmake-fail.txt"
  tmprunfailfile="results_report$2/tmpfailing-tests.txt"

  passfile="results_report$2/passing-tests.txt"
  compfailfile="results_report$2/make-fail.txt"
  runfailfile="results_report$2/failing-tests.txt"

  rm -f $tmppassfile $tmpcompfailfile $tmprunfailfile
  local -n myarray=$1
  for i in "${!myarray[@]}"; do
    local val=${myarray[$i]}
    if [ "$val" == "PASSPASS" ]; then
      echo $i >> $tmppassfile
    elif [ "$val" == "FAIL" ]; then
      echo $i >> $tmpcompfailfile
    elif [ "$val" == "PASSFAIL" ]; then
      echo $i >> $tmprunfailfile
    fi
  done
  if [ -e "$tmppassfile" ]; then
    sort $tmppassfile > results_report$2/passing-tests.txt
  fi
  if [ -e "$tmpcompfailfile" ]; then
    sort $tmpcompfailfile > results_report$2/make-fail.txt
  fi
  if [ -e "$tmprunfailfile" ]; then
    sort $tmprunfailfile > results_report$2/failing-tests.txt
  fi
  rm -f $tmppassfile $tmpcompfailfile $tmprunfailfile
}
vers="45 50"
for ver in $vers; do
  declare -A results$ver
  parse results$ver "results_report$ver/results.json"
  checkresult results$ver $ver
done

exit
popd
