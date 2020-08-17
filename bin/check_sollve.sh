#!/bin/bash
#
#  check_sollve.sh
#  Assumes run_sollve.sh has been executed. To lock the sollve repo at a hash use ROCMASTER=1 ./run_sollve.sh
#

# Input file
file="combined-results.txt"
pushd $HOME/git/aomp-test/sollve_vv

# Regex for getting testname and compiler/runtime fail type
testname_regex='(.*\.cpp|.*\.c|.*\.F90).*\(([a-z]*)\)'

# Skip known failures for now
skip_tests='declare_target_module.F90 test_target_enter_data_if.F90 test_target_firstprivate.F90 test_target_private.F90 test_target_teams_distribute_default_firstprivate.F90 test_target_teams_distribute_reduction_min.F90 test_declare_target_nested.c test_declare_variant.c test_requires_dynamic_allocators.c test_target_data_use_device_addr.c test_target_mapping_before_alloc.c test_target_mapping_before_delete.c test_target_teams_distribute_reduction_bitand.F90 test_target_teams_distribute_reduction_bitor.c test_target_teams_distribute_reduction_min.F90 test_target_teams_distribute_reduction_sub.F90 test_teams_distribute_default_none.c test_target_teams_distribute_defaultmap.F90 test_target_data_map_devices.F90 test_target_data_map_set_default_device.F90 test_target_enter_data_devices.F90 test_target_enter_data_set_default_device.F90 test_target_enter_exit_data_devices.F90 test_target_teams_distribute_device.F90'

total_fails=0
while read -r line; do
  # If line in file matches a testname and fail type save the info
  if [[ "$line" =~ $testname_regex ]]; then
    sollve_test=${BASH_REMATCH[1]}
    failure_type=${BASH_REMATCH[2]}
  fi
  if [[ $sollve_test != "" ]] ; then
     skip=1
     if [[ "$failure_type" == "compiler" ]] ; then
      # If failing tests is not in skip_tests, add to compiler unexpected failures.
      if [[ ! $skip_tests =~ $sollve_test ]]; then
        compile_fails+="$sollve_test "
        skip=0
      fi
    fi
    if [[ "$failure_type" == "runtime" ]] ; then
      # If failing tests is not in skip_tests, add to runtime unexpected failures.
      if [[ ! $skip_tests =~ $sollve_test ]]; then
        runtime_fails+="$sollve_test "
        skip=0
      fi
    fi
    if [[ "$failure_type" == "compiler" ]] || [[ $failure_type == "runtime" ]] && [[ $skip == 0 ]]; then
      ((total_fails+=1))
    fi
    sollve_test=""
    failure_type=""
  fi
done < "$file"
echo " "
if [[ "$runtime_fails" != "" ]]; then
  echo "Runtime Fails: "
  for fail in $runtime_fails; do
    echo "$fail"
  done
fi
echo ""
echo ""
if [[ "$compile_fails" != "" ]]; then
  echo "Compile Fails:"
  for fail in $compile_fails; do
    echo "$fail"
  done
  echo " "
fi
echo "Total Failures: $total_fails"
popd
