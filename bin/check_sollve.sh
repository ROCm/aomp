#!/bin/bash
#
#  check_sollve.sh
#

#Input file
file="results.json"
pushd $HOME/git/aomp-test/sollve_vv/results_report

#Regex for results json
testname_regex='Test name":."(.*)"'
compile_regex='Compiler result": "(.*)"'
runtime_regex='Runtime result": "(.*)"'

#skip known failures for now
skip_tests='test_target_enter_data_classes_inheritance.cpp test_target_enter_exit_data_classes.cpp declare_target_module.F90 test_target_firstprivate.F90 test_target_private.F90 test_target_teams_distribute_firstprivate.F90 test_target_teams_distribute_reduction_min.F90 test_target_teams_distribute_default_firstprivate.F90'

total_fails=0
while read -r line; do
  if [[ "$line" =~ $testname_regex ]]; then
    sollve_test=${BASH_REMATCH[1]}
  elif [[ "$line" =~ $compile_regex ]]; then
    compile_result=${BASH_REMATCH[1]}
  elif [[ "$line" =~ $runtime_regex ]]; then
    runtime_result=${BASH_REMATCH[1]}
  fi
  if [[ $sollve_test != "" ]] ; then
     skip=1
     if [[ $compile_result == "FAIL" ]] ; then
      if [[ ! $skip_tests =~ $sollve_test ]]; then
        compile_fails+="$sollve_test "
        skip=0
      fi
    fi
    if [[ $runtime_result == "FAIL" ]] ; then
      if [[ ! $skip_tests =~ $sollve_test ]]; then
        runtime_fails+="$sollve_test "
        skip=0
      fi
    fi
    if [[ $runtime_result == "FAIL" ]] || [[ $compile_result == "FAIL" ]] && [[ $skip == 0 ]]; then
      ((total_fails+=1))
    fi
    sollve_test=""
    compile_result=""
    runtime_result=""
  fi
done < "$file"
echo " "
echo "Runtime Fails: "
for fail in $runtime_fails; do
  echo "$fail"
done
echo ""
echo ""
echo "Compile Fails: $compile_fails"
for fail in $compile_fails; do
  echo "$fail"
done
echo " "
echo "Total Failures: $total_fails"
popd
