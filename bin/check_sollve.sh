#!/bin/bash
#
#  check_sollve.sh
#  Assumes run_sollve.sh has been executed.
#
set -e

declare -A results45
declare -A results50

pushd $HOME/git/aomp-test/sollve_vv
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
      if [ "$1" == 45 ]; then
        results45[$sollvetest]="$compresult"
        results45[$sollvetest]+="$runresult"
      elif [ "$1" == 50 ]; then
        results50[$sollvetest]="$compresult"
        results50[$sollvetest]+="$runresult"
      else
        echo "Unsupported value: $1"
        exit 1
     fi
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

  if [ "$1" == 45 ]; then
    for i in "${!results45[@]}"; do
      local val=${results45[$i]}
      checkstatus $1 $val
    done
  elif [ "$1" == 50 ]; then
    for i in "${!results50[@]}"; do
      local val=${results50[$i]}
      checkstatus $1 $val
    done
    else
      echo "Unsupported value: $1"
      exit 1
   fi

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
vers="45 50"
for ver in $vers; do
  parse $ver "results_report$ver/results.json"
  checkresult $ver
done

exit
popd
