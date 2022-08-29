
#define _XTEAM_NUM_THREADS 1024

// Define the base test class Tc
template <class T>
class Tc_complex {
  public:
    virtual ~Tc_complex(){}
    virtual T omp_dot() = 0;
    virtual T sim_dot() = 0;
    virtual void init_arrays(T initA, T initB, T initC) = 0;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) = 0;
};


// Clang Codegen needs to generate these declares
#if defined(__AMDGCN__) || defined(__NVPTX__)
  //  Headers for reduction helpers in DeviceRTLs/src/Reduction.cpp
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteamr_sum_cd(
		  double _Complex,double _Complex*,double _Complex*, unsigned int*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteamr_sum_cf(
		  float _Complex,float _Complex*,float _Complex*, unsigned int*);
#else
  // host variants are needed only for host fallback of simulated codegen functions sim_*
  // Dont bother making these correct
  extern "C"  void __kmpc_xteamr_sum_cd(double _Complex val, double _Complex* dval, 
		  double _Complex* xteam_mem, unsigned int * teams_done_ptr) 
{ *dval = val;}
  extern "C"  void __kmpc_xteamr_sum_cf(float _Complex val, float _Complex*rval,
		  float _Complex* xteam_mem, unsigned int * teams_done_ptr) 
{ *rval = val;}
#endif

// Overloaded functions used in simulated reductions below
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum_c(double _Complex val, 
		double _Complex* rval, double _Complex* xteam_mem, unsigned int * teams_done_ptr) {
  __kmpc_xteamr_sum_cd(val,rval,xteam_mem, teams_done_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum_c(float _Complex val, 
		float _Complex* rval, float _Complex* xteam_mem, unsigned int * teams_done_ptr) {
  __kmpc_xteamr_sum_cf(val,rval,xteam_mem, teams_done_ptr);
}

template <class T>
class ReductionsTestClass_complex : public Tc_complex<T> {
  protected:
    int array_size;
    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    ReductionsTestClass_complex(const int ARRAY_SIZE) {
      array_size = ARRAY_SIZE;
      // Allocate on the host
      a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
      b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
      c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
      #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
      {}
    }
    ~ReductionsTestClass_complex() {
      T *a = this->a;
      T *b = this->b;
      T *c = this->c;
      // End data region on device
      #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
      {}
      free(a);
      free(b);
      free(c);
    }

    void init_arrays(T initA, T initB, T initC) {
      T *a = this->a;
      T *b = this->b;
      T *c = this->c;
      //#pragma omp target teams distribute parallel for
      for (int i = 0; i < array_size; i++) {
        a[i] = initA;
        b[i] = initB;
        c[i] = initC;
      }
    }

    void read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c) {
      T *a = this->a;
      T *b = this->b;
      T *c = this->c;
      #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
      {}
      //#pragma omp parallel for
      for (int i = 0; i < array_size; i++) {
        h_a[i] = a[i];
        h_b[i] = b[i];
        h_c[i] = c[i];
      }
    }

    T omp_dot() {
      T sum;
      __real__(sum)  = 0.0;
      __imag__(sum)  = 0.0;
      T *a = this->a;
      T *b = this->b;
      #pragma omp target teams distribute parallel for map(tofrom: sum) reduction(+:sum)
      for (int i = 0; i < array_size; i++)
        sum += a[i] * b[i];
      return sum;
    }
 
    T sim_dot() {
      T sum;
      __real__(sum)  = 0.0;
      __imag__(sum)  = 0.0;
      T *a = this->a;
      T *b = this->b;
      int devid = 0; // use default or devid specified by user. 
      int team_procs = ompx_get_team_procs(0);

      // Need pinned, xteam_mem, and teams_done  or each reduction. Multiple reductions may
      // have different types, so need different target memory (global) allocations.
      T * pinned_sum0 = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      T * xteam_mem0 = (T *)omp_target_alloc(sizeof(T) * team_procs,devid);
      // teams_done0 is initialized here on host then copied 
      // to device as a global variable on host for atomic increment.
      unsigned int teams_done0 = 0;
      *pinned_sum0 = sum ;

      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:pinned_sum0[0:1]) map(to:teams_done0) is_device_ptr(xteam_mem0)
         // Repeat above 2 maps and is_device_ptr for each reduction
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
	// Repeat below for each reduction. 
        T val0;
        #pragma omp allocate(val0) allocator(omp_thread_mem_alloc)
        //val0 = T(0);
        __real__(val0)  = 0.0;
        __imag__(val0)  = 0.0;
        for (unsigned int i = k; i < array_size ; i += team_procs*_XTEAM_NUM_THREADS)
          val0 += a[i] * b[i];

        __kmpc_xteamr_sum_c(val0, &pinned_sum0[0], xteam_mem0, &teams_done0);
      }
      sum = *pinned_sum0;
      omp_free(pinned_sum0);
      omp_target_free(xteam_mem0,devid);
      return sum;
    }

};
