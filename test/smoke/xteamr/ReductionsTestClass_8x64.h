
#define _XTEAM_NUM_THREADS 512

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
void __kmpc_xteamr_sum_d_8x64(double, double *, double *,  unsigned int *);
void __kmpc_xteamr_sum_f_8x64(float, float *, float *,  unsigned int *);
void __kmpc_xteamr_sum_i_8x64(int, int *, int *,  unsigned int *);
void __kmpc_xteamr_sum_ui_8x64(unsigned int, unsigned int *, unsigned int *,  unsigned int *);
void __kmpc_xteamr_sum_l_8x64(long int, long int *, long int *,  unsigned int *);
void __kmpc_xteamr_sum_ul_8x64(unsigned long , unsigned long *, unsigned long *,  unsigned int *);
void __kmpc_xteamr_max_d_8x64(double, double *, double *,  unsigned int *);
void __kmpc_xteamr_max_f_8x64(float, float *, float *,  unsigned int *);
void __kmpc_xteamr_max_i_8x64(int, int *, int *,  unsigned int *);
void __kmpc_xteamr_max_ui_8x64(unsigned int, unsigned int *, unsigned int *,  unsigned int *);
void __kmpc_xteamr_max_l_8x64(long int, long int *, long int *,  unsigned int *);
void __kmpc_xteamr_max_ul_8x64(unsigned long , unsigned long *, unsigned long *,  unsigned int *);
void __kmpc_xteamr_min_d_8x64(double, double *, double *,  unsigned int *);
void __kmpc_xteamr_min_f_8x64(float, float *, float *,  unsigned int *);
void __kmpc_xteamr_min_i_8x64(int, int *, int *,  unsigned int *);
void __kmpc_xteamr_min_ui_8x64(unsigned int, unsigned int *, unsigned int *,  unsigned int *);
void __kmpc_xteamr_min_l_8x64(long int, long int *, long int *,  unsigned int *);
void __kmpc_xteamr_min_ul_8x64(unsigned long , unsigned long *, unsigned long *,  unsigned int *);

#else
  // host variants are needed only for host fallback of simulated codegen functions sim_*
  // Dont bother making these definitions correct
void __kmpc_xteamr_sum_d_8x64(double val, double * pval, double * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_f_8x64(float val, float * pval, float * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_i_8x64(int val, int * pval, int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_ui_8x64(unsigned int val, unsigned int *pval, unsigned int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_l_8x64(long int val, long int * pval, long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_sum_ul_8x64(unsigned long val, unsigned long * pval, unsigned long * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_d_8x64(double val, double * pval , double * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_f_8x64(float val, float * pval, float * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_i_8x64(int val, int * pval, int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_ui_8x64(unsigned int val, unsigned int * pval, unsigned int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_l_8x64(long int val, long int * pval, long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_max_ul_8x64(unsigned long val, unsigned long * pval, unsigned long * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_d_8x64(double val, double * pval, double * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_f_8x64(float val, float * pval, float * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_i_8x64(int val, int * pval, int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_ui_8x64(unsigned int val, unsigned int * pval, unsigned int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_l_8x64(long int val, long int * pval, long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
void __kmpc_xteamr_min_ul_8x64(unsigned long val, unsigned long * pval, unsigned long int * xteam_mem, unsigned int * ptr)
  { *pval = val;}
#endif
}

// Overloaded functions used in simulated reductions below
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(double val, double* rval,
	       	double* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_sum_d_8x64(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(float val, float* rval,
	       	float* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_sum_f_8x64(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(int val, int* rval,
	       	int* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_sum_i_8x64(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(unsigned int val, unsigned int* rval,
	       	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_sum_ui_8x64(val,rval,xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(long val, long* rval,
	       	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_sum_l_8x64(val,rval, xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(unsigned long val, unsigned long * rval,
	       	unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_sum_ul_8x64(val,rval, xteam_mem,td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(double val, double* rval,
	double * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_max_d_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(float val, float* rval,
	float * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_max_f_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(int val, int* rval,
	int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_max_i_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_max_ui_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(long val, long * rval,
	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_max_l_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(unsigned long val, unsigned long * rval,
	unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_max_ul_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(double val, double* rval,
	double * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_min_d_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(float val, float* rval,
	float * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_min_f_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(int val, int* rval,
	int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_min_i_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_min_ui_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(long val, long * rval,
	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_min_l_8x64(val,rval,xteam_mem, td_ptr);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(unsigned long val, unsigned long * rval,
	 unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_min_ul_8x64(val,rval,xteam_mem, td_ptr);
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
      //printf(" ====> omp_max max: %p array_size:%p \n",&maxval, &array_size); 
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
      int devid =  0;
      static uint32_t * teams_done_ptr0 = nullptr;
      static uint32_t * d_teams_done_ptr0;
      static T* d_team_vals0;
      static uint32_t team_procs0;
      if ( !teams_done_ptr0 ) {
         // One-time alloc device array for each teams's reduction value.
         team_procs0 = ompx_get_team_procs(devid);
         d_team_vals0  = (T *) omp_target_alloc(sizeof(T) * team_procs0, devid);
	 // Allocate and copy the zero-initialized teams_done counter one time
	 // because it atomically resets when last team increments it.
         teams_done_ptr0 = (uint32_t *)  malloc(sizeof(uint32_t));
         *teams_done_ptr0 = 0;
         d_teams_done_ptr0 = (uint32_t *) omp_target_alloc(sizeof(uint32_t),devid);
         omp_target_memcpy(d_teams_done_ptr0, teams_done_ptr0, 
	     sizeof(uint32_t), 0, 0, devid, omp_get_initial_device());
      }
      // Making the array_size 64 bits avoids a data_submit and data_retrieve 
      const uint64_t team_procs = team_procs0;
      const uint64_t as64 = (uint64_t) array_size;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:sum) is_device_ptr(d_team_vals0,d_teams_done_ptr0)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val0 = T(0);
        for (unsigned int i = k; i < as64 ; i += team_procs*_XTEAM_NUM_THREADS) {
          val0 += a[i] * b[i];
	}
        __kmpc_xteamr_sum(val0, &sum, d_team_vals0, d_teams_done_ptr0);
      }
      return sum;
    }

    T sim_max() {
      T *a = this->a;
      T *b = this->b;
      int devid =  0;
      T minval = std::numeric_limits<T>::lowest();
      T retval = minval;
      static uint32_t * teams_done_ptr1 = nullptr;
      static uint32_t * d_teams_done_ptr1;
      static T* d_team_vals1;
      static uint32_t team_procs1;
      if ( !teams_done_ptr1 ) {
         // One-time alloc device array for each teams's reduction value.
         team_procs1 = ompx_get_team_procs(devid);
         d_team_vals1  = (T *) omp_target_alloc(sizeof(T) * team_procs1, devid);
	 // Allocate and copy the zero-initialized teams_done counter one time
	 // because it atomically resets when last team increments it.
         teams_done_ptr1 = (uint32_t *)  malloc(sizeof(uint32_t));
         *teams_done_ptr1 = 0;
         d_teams_done_ptr1 = (uint32_t *) omp_target_alloc(sizeof(uint32_t),devid);
         omp_target_memcpy(d_teams_done_ptr1, teams_done_ptr1, 
	     sizeof(uint32_t), 0, 0, devid, omp_get_initial_device());
      }
      // Making the array_size 64 bits somehow avoids a data_submit and data_retrieve.?
      const uint64_t team_procs = team_procs1;
      const uint64_t as64 = (uint64_t) array_size;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:retval) is_device_ptr(d_team_vals1,d_teams_done_ptr1)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val1 = retval;
        for (unsigned int i = k; i < as64 ; i += team_procs*_XTEAM_NUM_THREADS){
	  val1 = (c[i] > val1) ? c[i] : val1;
	}
        __kmpc_xteamr_max(val1, &retval, d_team_vals1, d_teams_done_ptr1);
      }
      return retval;
    }

    T sim_min() {
      T *a = this->a;
      T *b = this->b;
      int devid =  0;
      T maxval = std::numeric_limits<T>::max();
      T retval = maxval;
      static uint32_t * teams_done_ptr2;
      static uint32_t * d_teams_done_ptr2;
      static T* d_team_vals2;
      static uint32_t team_procs2;
      if ( !teams_done_ptr2 ) {
         // One-time alloc device array for each teams's reduction value.
         team_procs2 = ompx_get_team_procs(devid);
         d_team_vals2  = (T *) omp_target_alloc(sizeof(T) * team_procs2, devid);
	 // Allocate and copy the zero-initialized teams_done counter one time
	 // because it atomically resets when last team increments it.
         teams_done_ptr2 = (uint32_t *)  malloc(sizeof(uint32_t));
         *teams_done_ptr2 = 0;
         d_teams_done_ptr2 = (uint32_t *) omp_target_alloc(sizeof(uint32_t),devid);
         omp_target_memcpy(d_teams_done_ptr2, teams_done_ptr2, 
	     sizeof(uint32_t), 0, 0, devid, omp_get_initial_device());
      }
      // Making the array_size 64 bits avoids a data_submit and data_retrieve.
      const uint64_t team_procs = team_procs2;
      const uint64_t as64 = (uint64_t) array_size;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:retval) is_device_ptr(d_team_vals2,d_teams_done_ptr2)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val2 = retval;
        for (unsigned int i = k; i < as64 ; i += team_procs*_XTEAM_NUM_THREADS){
	  val2 = (c[i] < val2) ? c[i] : val2;
	}
        __kmpc_xteamr_min(val2, &retval, d_team_vals2, d_teams_done_ptr2);
      }
      return retval;
    }

};
