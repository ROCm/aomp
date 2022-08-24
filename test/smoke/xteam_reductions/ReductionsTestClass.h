
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


// Clang Codegen needs to generate these declares
#if defined(__AMDGCN__) || defined(__NVPTX__)
  //  Headers for reduction helpers in DeviceRTLs/src/Reduction.cpp
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_sum_d(double,double*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_sum_f(float,float*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_sum_i(int,int*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_max_d(double,double*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_max_f(float,float*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_max_i(int,int*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_min_d(double,double*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_min_f(float,float*);
  extern "C" __attribute__((flatten, always_inline)) void __kmpc_xteam_min_i(int,int*);
#else
  // host variants are needed only for host fallback of simulated codegen functions sim_*
  // Dont bother making these correct
  extern "C"  void __kmpc_xteam_sum_d(double val, double* dval) { *dval = val;}
  extern "C"  void __kmpc_xteam_sum_f(float val, float*rval) { *rval = val;}
  extern "C"  void __kmpc_xteam_sum_i(int val, int*rval) { *rval = val;}
  extern "C"  void __kmpc_xteam_max_d(double val, double* dval) { *dval = val;}
  extern "C"  void __kmpc_xteam_max_f(float val, float*rval) { *rval = val;}
  extern "C"  void __kmpc_xteam_max_i(int val, int*rval) { *rval = val;}
  extern "C"  void __kmpc_xteam_min_d(double val, double* dval) { *dval = val;}
  extern "C"  void __kmpc_xteam_min_f(float val, float*rval) { *rval = val;}
  extern "C"  void __kmpc_xteam_min_i(int val, int*rval) { *rval = val;}
#endif

// Overloaded functions used in simulated reductions below
void __attribute__((flatten, always_inline)) __kmpc_xteam_sum(double val, double* rval) {
  __kmpc_xteam_sum_d(val,rval);
}
void __attribute__((flatten, always_inline)) __kmpc_xteam_sum(float val, float* rval) {
  __kmpc_xteam_sum_f(val,rval);
}
void __attribute__((flatten, always_inline)) __kmpc_xteam_sum(int val, int* rval) {
  __kmpc_xteam_sum_i(val,rval);
}

void __attribute__((flatten, always_inline)) __kmpc_xteam_max(double val, double* rval) {
  __kmpc_xteam_max_d(val,rval);
}
void __attribute__((flatten, always_inline)) __kmpc_xteam_max(float val, float* rval) {
  __kmpc_xteam_max_f(val,rval);
}
void __attribute__((flatten, always_inline)) __kmpc_xteam_max(int val, int* rval) {
  __kmpc_xteam_max_i(val,rval);
}

void __attribute__((flatten, always_inline)) __kmpc_xteam_min(double val, double* rval) {
  __kmpc_xteam_min_d(val,rval);
}
void __attribute__((flatten, always_inline)) __kmpc_xteam_min(float val, float* rval) {
  __kmpc_xteam_min_f(val,rval);
}
void __attribute__((flatten, always_inline)) __kmpc_xteam_min(int val, int* rval) {
  __kmpc_xteam_min_i(val,rval);
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
      T sum1 = T(0);
      T *a = this->a;
      T *b = this->b;
      T * pinned_sum = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      T * pinned_sum1 = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      pinned_sum[0] = sum ;
      pinned_sum1[0] = sum1 ;
      int team_procs = ompx_get_team_procs(0);
#pragma omp target teams distribute parallel for map(from:pinned_sum[0:1], pinned_sum1[0:1] ) num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS)
      for (unsigned int k=0; k<(team_procs*1024); k++) {
        T val;
	T val1;
        #pragma omp allocate(val) allocator(omp_thread_mem_alloc)
        val = T(0);
        #pragma omp allocate(val1) allocator(omp_thread_mem_alloc)
        val1 = T(0);
        for (unsigned int i = k; i < array_size ; i += team_procs*1024) {
          val += a[i] * b[i];
          val1 += a[i] * b[i];
	}
        __kmpc_xteam_sum(val,&pinned_sum[0]);
        __kmpc_xteam_sum(val1,&pinned_sum1[0]);
      }
      sum = pinned_sum[0];
      sum1 = pinned_sum1[0];
      printf("sum=%f sum1=%f\n", sum, sum1);
      omp_free(pinned_sum);
      omp_free(pinned_sum1);
      return sum;
    }

    T sim_max() {
      T *c = this->c;
      T maxval = std::numeric_limits<T>::lowest(); 
      T * pinned_max = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      pinned_max[0] = maxval;
      int team_procs = ompx_get_team_procs(0);
      #pragma omp target teams distribute parallel for map(from:pinned_max[0:1]) num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS)
      for (unsigned int k=0; k<(team_procs*1024); k++) {
        T val;
        #pragma omp allocate(val) allocator(omp_thread_mem_alloc)
	val = maxval ; 
        for (unsigned int i = k; i < array_size ; i += team_procs*1024)
	  if (c[i] > val) val = c[i];

        __kmpc_xteam_max(val,&pinned_max[0]);
      }
      maxval = pinned_max[0];
      omp_free(pinned_max);
      return maxval;
    }

    T sim_min() {
      T minval = std::numeric_limits<T>::max();
      T *c = this->c;
      T * pinned_min = (T *)omp_alloc(sizeof(T), ompx_pinned_mem_alloc);
      pinned_min[0] = minval;
      int team_procs = ompx_get_team_procs(0);
      #pragma omp target teams distribute parallel for map(from:pinned_min[0:1]) num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS)
      for (unsigned int k=0; k<(team_procs*1024); k++) {
        T val;
        #pragma omp allocate(val) allocator(omp_thread_mem_alloc)
        val = minval ; 
        for (unsigned int i = k; i < array_size ; i += team_procs*1024)
          if (c[i] < val) val = c[i];

        __kmpc_xteam_min(val,&pinned_min[0]);
      }
      minval = pinned_min[0];
      omp_free(pinned_min);
      return minval;
   }
};
