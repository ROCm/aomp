
/// This could create a conflicting .omp_offloading.entry
void test_comp_unit_2(const int niters, double* a)
{
#pragma omp target
  for(int ii = 0; ii < niters; ++ii)
    a[ii] *= 2.0;
}
