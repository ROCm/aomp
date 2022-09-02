
#define _XTEAM_NUM_THREADS 1024

// Define the base test class Tc
template <class T>
class Tc {
  public:
    virtual ~Tc(){}
    virtual T omp_dot() = 0;
    virtual T sim_dot() = 0;
    virtual T omp_max() = 0;
    virtual T sim_max() = 0;
    virtual T omp_min() = 0;
    virtual T sim_min() = 0;
    virtual void init_arrays(T initA, T initB, T initC) = 0;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) = 0;
};


extern "C" {
// Clang Codegen needs to generate these declares
#if defined(__AMDGCN__) || defined(__NVPTX__)
  //  Headers for reduction helpers in DeviceRTLs/src/Reduction.cpp
void __kmpc_xteamr_sum_d(double, double *, double *, unsigned int *);
void __kmpc_xteamr_sum_f(float, float *, float *, unsigned int *);
void __kmpc_xteamr_sum_i(int, int *, int *, unsigned int *);
void __kmpc_xteamr_sum_ui(unsigned int, unsigned int *, unsigned int *, unsigned int *);
void __kmpc_xteamr_sum_l(long int, long int *, long int *, unsigned int *);
void __kmpc_xteamr_sum_ul(unsigned long , unsigned long *, unsigned long *, unsigned int *);
void __kmpc_xteamr_max_d(double, double *, double *, unsigned int *);
void __kmpc_xteamr_max_f(float, float *, float *, unsigned int *);
void __kmpc_xteamr_max_i(int, int *, int *, unsigned int *);
void __kmpc_xteamr_max_ui(unsigned int, unsigned int *, unsigned int *, unsigned int *);
void __kmpc_xteamr_max_l(long int, long int *, long int *, unsigned int *);
void __kmpc_xteamr_max_ul(unsigned long , unsigned long *, unsigned long *, unsigned int *);
void __kmpc_xteamr_min_d(double, double *, double *, unsigned int *);
void __kmpc_xteamr_min_f(float, float *, float *, unsigned int *);
void __kmpc_xteamr_min_i(int, int *, int *, unsigned int *);
void __kmpc_xteamr_min_ui(unsigned int, unsigned int *, unsigned int *, unsigned int *);
void __kmpc_xteamr_min_l(long int, long int *, long int *, unsigned int *);
void __kmpc_xteamr_min_ul(unsigned long , unsigned long *, unsigned long *, unsigned int *);

#else
  // host variants are needed only for host fallback of simulated codegen functions sim_*
  // Dont bother making these definitions correct
void __kmpc_xteamr_sum_d(double val, double * pval, double * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_f(float val, float * pval, float * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_i(int val, int * pval, int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_ui(unsigned int val, unsigned int *pval, unsigned int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_l(long int val, long int * pval, long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_ul(unsigned long val, unsigned long * pval, unsigned long * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_d(double val, double * pval , double * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_f(float val, float * pval, float * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_i(int val, int * pval, int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_ui(unsigned int val, unsigned int * pval, unsigned int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_l(long int val, long int * pval, long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_ul(unsigned long val, unsigned long * pval, unsigned long * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_d(double val, double * pval, double * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_f(float val, float * pval, float * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_i(int val, int * pval, int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_ui(unsigned int val, unsigned int * pval, unsigned int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_l(long int val, long int * pval, long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_ul(unsigned long val, unsigned long * pval, unsigned long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
#endif
}

// Overloaded functions used in simulated reductions below
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(double val, double* rval,
	       	double* xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_sum_d(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(float val, float* rval,
	       	float* xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_sum_f(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(int val, int* rval,
	       	int* xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_sum_i(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(unsigned int val, unsigned int* rval,
	       	unsigned int * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_sum_ui(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(long val, long* rval,
	       	long * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_sum_l(val,rval, xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(unsigned long val, unsigned long * rval,
	       	unsigned long * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_sum_ul(val,rval, xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(double val, double* rval,
	double * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_max_d(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(float val, float* rval,
	float * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_max_f(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(int val, int* rval,
	int * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_max_i(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_max_ui(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(long val, long * rval,
	long * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_max_l(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(unsigned long val, unsigned long * rval,
	unsigned long * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_max_ul(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(double val, double* rval,
	double * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_min_d(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(float val, float* rval,
	float * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_min_f(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(int val, int* rval,
	int * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_min_i(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_min_ui(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(long val, long * rval,
	long * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_min_l(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(unsigned long val, unsigned long * rval,
	 unsigned long * xteam_mem, unsigned int * td_ptr) {
  __kmpc_xteamr_min_ul(val,rval,xteam_mem, td_ptr);
}

template <class T>
class ReductionsTestClass : public Tc<T> {
  protected:
    int array_size;
    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    ReductionsTestClass(const int ARRAY_SIZE) {
      array_size = ARRAY_SIZE;
      // Allocate on the host
      a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
      b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
      c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
      #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
      {}
    }
    ~ReductionsTestClass() {
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
      #pragma omp target teams distribute parallel for
      for (int i = 0; i < array_size; i++) {
        a[i] = initA;
        b[i] = initB;
        c[i] = initC * (i+1);
      }
    }

    void read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c) {
      T *a = this->a;
      T *b = this->b;
      T *c = this->c;
      #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
      {}
      #pragma omp parallel for
      for (int i = 0; i < array_size; i++) {
        h_a[i] = a[i];
        h_b[i] = b[i];
        h_c[i] = c[i];
      }
    }

    T omp_dot() {
      T sum = 0.0;
      T *a = this->a;
      T *b = this->b;
      #pragma omp target teams distribute parallel for map(tofrom: sum) reduction(+:sum)
      for (int i = 0; i < array_size; i++)
        sum += a[i] * b[i];
      return sum;
    }

    T omp_max() {
      T maxval = std::numeric_limits<T>::lowest();
      T *c = this->c;
      #pragma omp target teams distribute parallel for map(tofrom: maxval) reduction(max:maxval)
      for (int i = 0; i < array_size; i++)
	maxval = (c[i] > maxval) ? c[i] : maxval;
      return maxval;
    }

    T omp_min() {
      T minval = std::numeric_limits<T>::max();
      T *c = this->c;
      #pragma omp target teams distribute parallel for map(tofrom: minval) reduction(min:minval)
      for (int i = 0; i < array_size; i++) {
	minval = (c[i] < minval) ? c[i] : minval;
      }
      return minval;
    }

    //  These simulations of reductions are what the optimized openmp comiler will codegen. 
 
    T sim_dot() {
      T sum = T(0);
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
        val0 = T(0);
        for (unsigned int i = k; i < array_size ; i += team_procs*_XTEAM_NUM_THREADS)
          val0 += a[i] * b[i];

        __kmpc_xteamr_sum(val0, &pinned_sum0[0], xteam_mem0, &teams_done0);
      }
      sum = *pinned_sum0;
      omp_free(pinned_sum0);
      omp_target_free(xteam_mem0,devid);
      return sum;
    }

    T sim_max() {
      T *c = this->c;
      T retval;
      int devid = 0; // use default or devid specified by user. 
      int team_procs = ompx_get_team_procs(0);
      T * pinned_max0 = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      T * xteam_mem0 = (T *)omp_target_alloc(sizeof(T) * team_procs,devid);
      // teams_done0 is initialized here on host then copied 
      // to device as a global variable on host for atomic increment  
      unsigned int teams_done0 = 0;
      T minval = std::numeric_limits<T>::lowest();
      *pinned_max0 = minval;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:pinned_max0[0:1]) map(to:teams_done0) is_device_ptr(xteam_mem0)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val0;
        #pragma omp allocate(val0) allocator(omp_thread_mem_alloc)
	val0 = minval ; 
        for (unsigned int i = k; i < array_size ; i += team_procs*_XTEAM_NUM_THREADS)
	  if (c[i] > val0) val0 = c[i];

        __kmpc_xteamr_max(val0,&pinned_max0[0],xteam_mem0, &teams_done0);
      }
      retval = *pinned_max0;
      omp_free(pinned_max0);
      omp_target_free(xteam_mem0,devid);
      return retval;
    }

    T sim_min() {
      T *c = this->c;
      T retval;
      int devid = 0; // use default or devid specified by user. 
      int team_procs = ompx_get_team_procs(0);
      T * pinned_min0 = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      T * xteam_mem0 = (T *)omp_target_alloc(sizeof(T) * team_procs,devid);
      // teams_done0 is initialized here on host then copied 
      // to device as a global variable on host for atomic increment  
      unsigned int teams_done0 = 0;
      T maxval = std::numeric_limits<T>::max();
      *pinned_min0 = maxval;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:pinned_min0[0:1]) map(to:teams_done0) is_device_ptr(xteam_mem0)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val0;
        #pragma omp allocate(val0) allocator(omp_thread_mem_alloc)
	val0 = maxval ; 
        for (unsigned int i = k; i < array_size ; i += team_procs*_XTEAM_NUM_THREADS)
	  if (c[i] < val0) val0 = c[i];

        __kmpc_xteamr_min(val0,&pinned_min0[0],xteam_mem0, &teams_done0);
      }
      retval = *pinned_min0;
      omp_free(pinned_min0);
      omp_target_free(xteam_mem0,devid);
      return retval;
    }

#if 0
    T sim_min() {
      T minval = std::numeric_limits<T>::max();
      T *c = this->c;
      T * pinned_min = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      pinned_min[0] = minval;
      int team_procs = ompx_get_team_procs(0);
      #pragma omp target teams distribute parallel for map(from:pinned_min[0:1]) num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val;
        #pragma omp allocate(val) allocator(omp_thread_mem_alloc)
        val = minval ; 
        for (unsigned int i = k; i < array_size ; i += team_procs*_XTEAM_NUM_THREADS)
          if (c[i] < val) val = c[i];

        __kmpc_xteamr_min(val,&pinned_min[0]);
      }
      minval = pinned_min[0];
      omp_free(pinned_min);
      return minval;
   }
#endif
};
