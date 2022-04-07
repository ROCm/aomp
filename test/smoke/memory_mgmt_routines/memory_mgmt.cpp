#include <iostream>
#include <vector>

#include <omp.h>

#define N 10000
#define ALIGN 16
#define KB 1024

void test_allocation_routines(omp_allocator_handle_t allocator);

template<typename T>
class trait_it
{
public:
  std::vector<T> t_arr;
  omp_memspace_handle_t m;
  omp_alloctrait_key_t t;
  trait_it(omp_memspace_handle_t m, omp_alloctrait_key_t t, std::vector<T> t_arr) : m(m), t(t), t_arr(t_arr) {}

  void test_all() {
    for (auto s : t_arr) {
      omp_alloctrait_t arr[1];
      arr[0].key = t;
      arr[0].value = s;
      auto allocator = omp_init_allocator(m, 1, arr);
      test_allocation_routines(allocator);
      omp_destroy_allocator(allocator);
    }
  }
};

int main() {
  // test all predefined memory spaces (OpenMP 5.1 specs, section 2.13.1)
  omp_memspace_handle_t omp_memspace_arr[] = {
    omp_default_mem_space,
    omp_large_cap_mem_space,
    omp_const_mem_space,
    omp_high_bw_mem_space,
    omp_low_lat_mem_space
  };

  omp_alloctrait_key_t omp_memalloc_traits[] = {
    omp_atk_sync_hint,
    omp_atk_alignment,
    omp_atk_access,
    omp_atk_pool_size,
    omp_atk_fallback,
    omp_atk_fb_data,
    omp_atk_pinned,
    omp_atk_partition
  };

  for(auto mem : omp_memspace_arr)
    for(auto trait : omp_memalloc_traits) {
      switch(trait) {
      case omp_atk_sync_hint: {
	std::vector<omp_alloctrait_value_t> omp_synchint_vals = {
	  omp_atv_contended,
	  omp_atv_uncontended,
	  omp_atv_serialized,
	  omp_atv_private
	};
	trait_it<omp_alloctrait_value_t> itt(mem, trait, omp_synchint_vals);
	itt.test_all();
	break;
      }
      case omp_atk_alignment: {
	std::vector<int> align_vals = {2, 4, 8, 16, 32, 64};
	trait_it<int> itt(mem, trait, align_vals);
	itt.test_all();
	break;
      }
      case omp_atk_access: {
	std::vector<omp_alloctrait_value_t> omp_access_vals = {
	  omp_atv_all,
	  omp_atv_thread,
	  omp_atv_pteam,
	  omp_atv_cgroup
	};
	trait_it<omp_alloctrait_value_t> itt(mem, trait, omp_access_vals);
	itt.test_all();
	break;
      }
      case omp_atk_pool_size: {
	std::vector<int> pool_sizes = {512*KB, 1024*KB, 2048*KB, 4096*KB};
	trait_it<int> itt(mem, trait, pool_sizes);
	itt.test_all();
	break;
      }
      case omp_atk_fallback: {
	std::vector<omp_alloctrait_value_t> omp_fallback_vals = {
	  omp_atv_default_mem_fb,
	  omp_atv_null_fb,
	  omp_atv_abort_fb,
	  //omp_atv_allocator_fb // not tested
	};
	trait_it<omp_alloctrait_value_t> itt(mem, trait, omp_fallback_vals);
	itt.test_all();
	break;
      }
      case omp_atk_pinned: {
	std::vector<omp_alloctrait_value_t> omp_pinned_vals = {
	  omp_atv_true,
	  omp_atv_false
	};
	trait_it<omp_alloctrait_value_t> itt(mem, trait, omp_pinned_vals);
	itt.test_all();
	break;
      }
      case omp_atk_partition: {
	std::vector<omp_alloctrait_value_t> omp_partition_vals = {
	    omp_atv_environment,
	    omp_atv_nearest,
	    omp_atv_blocked,
	    omp_atv_interleaved
	};
	trait_it<omp_alloctrait_value_t> itt(mem, trait, omp_partition_vals);
	itt.test_all();
	break;
      }
      default: {}
      }
    }
  return 0;
}

void check_alloc(void *ptr, const char call_name[]) {
  if(!ptr) {
    printf("Error in %s, returned nullptr\n", call_name);
    exit(1);
  }
}

void target_region_init_and_check(int *a, int size) {
  #pragma omp target teams distribute parallel for map(from:a[:size])
  for(int i = 0 ; i < size; i++)
    a[i] = i;

  for(int i = 0; i < size; i++)
    if(a[i] != i) {
      printf("Error at %d: tgt value is %d, expected value is %d\n", i, a[i], i);
      exit(1);
    }
}

void test_allocation_routines(omp_allocator_handle_t allocator) {
  int *a = (int *)omp_alloc(N*sizeof(int), allocator);
  check_alloc(a, "omp_alloc");
  target_region_init_and_check(a, N);
  omp_free(a, allocator);

  int *a_align = (int *)omp_aligned_alloc(ALIGN, N*sizeof(int), allocator);
  check_alloc(a_align, "omp_aligned_alloc");
  target_region_init_and_check(a_align, N);
  omp_free(a_align, allocator);

  int *a_calloc = (int *)omp_calloc(N, sizeof(int), allocator);
  check_alloc(a_calloc, "omp_calloc");
  target_region_init_and_check(a_calloc, N);
  omp_free(a_calloc, allocator);

  // OMP 5.1 call not yet available in openmp runtime
  // int *a_aligned_calloc = (int *)omp_aligned_calloc(ALIGN, N*sizeof(int), allocator);
  // omp_free(a_aligned_calloc, allocator);

  // OMP 5.1 call not yet available in openmp runtime
  int *a_realloc = (int *)omp_alloc(N*sizeof(int), allocator);
  check_alloc(a_realloc, "omp_alloc preceding omp_realloc");
  auto reallocator = omp_init_allocator(omp_default_mem_space, 0, {});
  a_realloc = (int *)omp_realloc(a_realloc, 2*N*sizeof(int), reallocator, allocator);
  check_alloc(a, "omp_realloc");
  target_region_init_and_check(a_realloc, N);
  omp_free(a_realloc, reallocator);
}
