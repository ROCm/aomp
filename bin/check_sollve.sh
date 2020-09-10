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
skip_tests='test_target_data_map_to_array_sections.F90 declare_target_module.F90 test_target_enter_data_if.F90 test_target_firstprivate.F90 test_target_private.F90 test_target_teams_distribute_default_firstprivate.F90 test_target_teams_distribute_reduction_min.F90 test_declare_target_nested.c test_declare_variant.c test_requires_dynamic_allocators.c test_target_data_use_device_addr.c test_target_mapping_before_alloc.c test_target_mapping_before_delete.c test_target_teams_distribute_reduction_bitand.F90 test_target_teams_distribute_reduction_bitor.c test_target_teams_distribute_reduction_min.F90 test_target_teams_distribute_reduction_sub.F90 test_teams_distribute_default_none.c test_target_teams_distribute_defaultmap.F90 test_target_data_map_devices.F90 test_target_data_map_set_default_device.F90 test_target_enter_data_devices.F90 test_target_enter_data_set_default_device.F90 test_target_enter_exit_data_devices.F90 test_target_teams_distribute_device.F90'

if [ "$EPSDB" == "1" ]; then
  skip_tests+=" test_target_depends.c test_target_enter_data_depend.c test_target_enter_exit_data_depend.c test_target_teams_distribute_depend_array_section.c test_target_teams_distribute_depend_disjoint_section.c test_target_teams_distribute_depend_in_in.c test_target_teams_distribute_depend_in_out.c test_target_teams_distribute_depend_list.c test_target_teams_distribute_depend_out_in.c test_target_teams_distribute_depend_out_out.c test_target_teams_distribute_depend_unused_data.c test_target_update_depend.c"
fi

# Additional failures seen with updated sollve repo as of 9/10/20
skip_tests+=" test_target_update_mapper_to_discontiguous.c test_declare_mapper_target_struct.c test_requires_unified_shared_memory.c test_requires_unified_shared_memory_heap.c test_requires_unified_shared_memory_heap_is_device_ptr.c test_requires_unified_shared_memory_heap_map.c test_requires_unified_shared_memory_omp_target_alloc.c test_requires_unified_shared_memory_omp_target_alloc_is_device_ptr.c test_requires_unified_shared_memory_stack.c test_requires_unified_shared_memory_stack_is_device_ptr.c test_requires_unified_shared_memory_stack_map.c test_requires_unified_shared_memory_static.c test_requires_unified_shared_memory_static_is_device_ptr.c test_requires_unified_shared_memory_static_map.c"

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
