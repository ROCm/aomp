
#define _XTEAM_NUM_THREADS 256

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
  //  Headers for reduction helpers in DeviceRTLs/include/Interface.h
#define _RF_LDS volatile __attribute__((address_space(3)))
void __kmpc_rfun_sum_d(double * val, double otherval);
void __kmpc_rfun_sum_lds_d(_RF_LDS double * val, _RF_LDS double * otherval);
void __kmpc_rfun_sum_f(float * val, float otherval);
void __kmpc_rfun_sum_lds_f(_RF_LDS float * val, _RF_LDS float * otherval);
void __kmpc_rfun_sum_i(int * val, int otherval);
void __kmpc_rfun_sum_lds_i(_RF_LDS int * val, _RF_LDS int * otherval);
void __kmpc_rfun_sum_ui(unsigned int * val, unsigned int otherval);
void __kmpc_rfun_sum_lds_ui(_RF_LDS unsigned int * val, _RF_LDS unsigned int * otherval);
void __kmpc_rfun_sum_l(long * val, long otherval);
void __kmpc_rfun_sum_lds_l(_RF_LDS long * val, _RF_LDS long * otherval);
void __kmpc_rfun_sum_ul(unsigned long * val, unsigned long otherval);
void __kmpc_rfun_sum_lds_ul(_RF_LDS unsigned long * val, _RF_LDS unsigned long * otherval);
void __kmpc_rfun_min_d(double * val, double otherval);
void __kmpc_rfun_min_lds_d(_RF_LDS double * val, _RF_LDS double * otherval);
void __kmpc_rfun_min_f(float * val, float otherval);
void __kmpc_rfun_min_lds_f(_RF_LDS float * val, _RF_LDS float * otherval);
void __kmpc_rfun_min_i(int * val, int otherval);
void __kmpc_rfun_min_lds_i(_RF_LDS int * val, _RF_LDS int * otherval);
void __kmpc_rfun_min_ui(unsigned int * val, unsigned int otherval);
void __kmpc_rfun_min_lds_ui(_RF_LDS unsigned int * val, _RF_LDS unsigned int * otherval);
void __kmpc_rfun_min_l(long * val, long otherval);
void __kmpc_rfun_min_lds_l(_RF_LDS long * val, _RF_LDS long * otherval);
void __kmpc_rfun_min_ul(unsigned long * val, unsigned long otherval);
void __kmpc_rfun_min_lds_ul(_RF_LDS unsigned long * val, _RF_LDS unsigned long * otherval);
void __kmpc_rfun_max_d(double * val, double otherval);
void __kmpc_rfun_max_lds_d(_RF_LDS double * val, _RF_LDS double * otherval);
void __kmpc_rfun_max_f(float * val, float otherval);
void __kmpc_rfun_max_lds_f(_RF_LDS float * val, _RF_LDS float * otherval);
void __kmpc_rfun_max_i(int * val, int otherval);
void __kmpc_rfun_max_lds_i(_RF_LDS int * val, _RF_LDS int * otherval);
void __kmpc_rfun_max_ui(unsigned int * val, unsigned int otherval);
void __kmpc_rfun_max_lds_ui(_RF_LDS unsigned int * val, _RF_LDS unsigned int * otherval);
void __kmpc_rfun_max_l(long * val, long otherval);
void __kmpc_rfun_max_lds_l(_RF_LDS long * val, _RF_LDS long * otherval);
void __kmpc_rfun_max_ul(unsigned long * val, unsigned long otherval);
void __kmpc_rfun_max_lds_ul(_RF_LDS unsigned long * val, _RF_LDS unsigned long * otherval);

void __kmpc_xteamr_d_4x64(double v, double *r_ptr, double *tvals,
                           uint32_t *td_ptr, void (*_rf)(double *, double),
                           void (*_rf_lds)(_RF_LDS double *,
                                           _RF_LDS double *),
                           double iv);
void __kmpc_xteamr_f_4x64(float v, float *r_ptr, float *tvals,
                           uint32_t *td_ptr, void (*_rf)(float *, float),
                           void (*_rf_lds)(_RF_LDS float *,
                                           _RF_LDS float *),
                           float iv);
void __kmpc_xteamr_i_4x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                           void (*_rf)(int *, int),
                           void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                           int iv);
void __kmpc_xteamr_ui_4x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                            void (*_rf_lds)(_RF_LDS uint32_t *,
                                            _RF_LDS uint32_t *),
                            uint32_t iv);
void __kmpc_xteamr_l_4x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                           void (*_rf)(long *, long),
                           void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                           long iv);
void __kmpc_xteamr_ul_4x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                            void (*_rf_lds)(_RF_LDS uint64_t *,
                                            _RF_LDS uint64_t *),
                            uint64_t iv);
#undef _RF_LDS


#else
  // host variants are needed only for host fallback of simulated codegen functions sim_*
  // Dont bother making these definitions correct

#define _RF_LDS
void __kmpc_rfun_sum_d(double * val, double otherval)
{ *val += otherval; }
void __kmpc_rfun_sum_lds_d(_RF_LDS double * val, _RF_LDS double * otherval)
{ *val += *otherval; }
void __kmpc_rfun_sum_f(float * val, float otherval)
{ *val += otherval; }
void __kmpc_rfun_sum_lds_f(_RF_LDS float * val, _RF_LDS float * otherval)
{ *val += *otherval; }
void __kmpc_rfun_sum_i(int * val, int otherval)
{ *val += otherval; }
void __kmpc_rfun_sum_lds_i(_RF_LDS int * val, _RF_LDS int * otherval)
{ *val += *otherval; }
void __kmpc_rfun_sum_ui(unsigned int * val, unsigned int otherval)
{ *val += otherval; }
void __kmpc_rfun_sum_lds_ui(_RF_LDS unsigned int * val, _RF_LDS unsigned int * otherval)
{ *val += *otherval; }
void __kmpc_rfun_sum_l(long * val, long otherval)
{ *val += otherval; }
void __kmpc_rfun_sum_lds_l(_RF_LDS long * val, _RF_LDS long * otherval)
{ *val += *otherval; }
void __kmpc_rfun_sum_ul(unsigned long * val, unsigned long otherval)
{ *val += otherval; }
void __kmpc_rfun_sum_lds_ul(_RF_LDS unsigned long * val, _RF_LDS unsigned long * otherval)
{ *val += *otherval; }

void __kmpc_rfun_min_d(double * val, double otherval)
{ *val = (otherval < *val) ? otherval : *val ; }
void __kmpc_rfun_min_lds_d(_RF_LDS double * val, _RF_LDS double * otherval)
{ *val = (*otherval < *val) ? *otherval : *val ; }
void __kmpc_rfun_min_f(float * val, float otherval)
{ *val = (otherval < *val) ? otherval : *val ; }
void __kmpc_rfun_min_lds_f(_RF_LDS float * val, _RF_LDS float * otherval)
{ *val = (*otherval < *val) ? *otherval : *val ; }
void __kmpc_rfun_min_i(int * val, int otherval)
{ *val = (otherval < *val) ? otherval : *val ; }
void __kmpc_rfun_min_lds_i(_RF_LDS int * val, _RF_LDS int * otherval)
{ *val = (*otherval < *val) ? *otherval : *val ; }
void __kmpc_rfun_min_ui(unsigned int * val, unsigned int otherval)
{ *val = (otherval < *val) ? otherval : *val ; }
void __kmpc_rfun_min_lds_ui(_RF_LDS unsigned int * val, _RF_LDS unsigned int * otherval)
{ *val = (*otherval < *val) ? *otherval : *val ; }
void __kmpc_rfun_min_l(long * val, long otherval)
{ *val = (otherval < *val) ? otherval : *val ; }
void __kmpc_rfun_min_lds_l(_RF_LDS long * val, _RF_LDS long * otherval)
{ *val = (*otherval < *val) ? *otherval : *val ; }
void __kmpc_rfun_min_ul(unsigned long * val, unsigned long otherval)
{ *val = (otherval < *val) ? otherval : *val ; }
void __kmpc_rfun_min_lds_ul(_RF_LDS unsigned long * val, _RF_LDS unsigned long * otherval)
{ *val = (*otherval < *val) ? *otherval : *val ; }

void __kmpc_rfun_max_d(double * val, double otherval)
{ *val = (otherval > *val) ? otherval : *val ; }
void __kmpc_rfun_max_lds_d(_RF_LDS double * val, _RF_LDS double * otherval)
{ *val = (*otherval > *val) ? *otherval : *val ; }
void __kmpc_rfun_max_f(float * val, float otherval)
{ *val = (otherval > *val) ? otherval : *val ; }
void __kmpc_rfun_max_lds_f(_RF_LDS float * val, _RF_LDS float * otherval)
{ *val = (*otherval > *val) ? *otherval : *val ; }
void __kmpc_rfun_max_i(int * val, int otherval)
{ *val = (otherval > *val) ? otherval : *val ; }
void __kmpc_rfun_max_lds_i(_RF_LDS int * val, _RF_LDS int * otherval)
{ *val = (*otherval > *val) ? *otherval : *val ; }
void __kmpc_rfun_max_ui(unsigned int * val, unsigned int otherval)
{ *val = (otherval > *val) ? otherval : *val ; }
void __kmpc_rfun_max_lds_ui(_RF_LDS unsigned int * val, _RF_LDS unsigned int * otherval)
{ *val = (*otherval > *val) ? *otherval : *val ; }
void __kmpc_rfun_max_l(long * val, long otherval)
{ *val = (otherval > *val) ? otherval : *val ; }
void __kmpc_rfun_max_lds_l(_RF_LDS long * val, _RF_LDS long * otherval)
{ *val = (*otherval > *val) ? *otherval : *val ; }
void __kmpc_rfun_max_ul(unsigned long * val, unsigned long otherval)
{ *val = (otherval > *val) ? otherval : *val ; }
void __kmpc_rfun_max_lds_ul(_RF_LDS unsigned long * val, _RF_LDS unsigned long * otherval)
{ *val = (*otherval > *val) ? *otherval : *val ; }

void __kmpc_xteamr_d_4x64(double v, double *r_ptr, double *tvals,
                           uint32_t *td_ptr, void (*_rf)(double *, double),
                           void (*_rf_lds)(_RF_LDS double *,
                                           _RF_LDS double *),
                           double iv)
  { *r_ptr = v;}
void __kmpc_xteamr_f_4x64(float v, float *r_ptr, float *tvals,
                           uint32_t *td_ptr, void (*_rf)(float *, float),
                           void (*_rf_lds)(_RF_LDS float *,
                                           _RF_LDS float *),
                           float iv)
  { *r_ptr = v;}
void __kmpc_xteamr_i_4x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                           void (*_rf)(int *, int),
                           void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                           int iv)
  { *r_ptr = v;}
void __kmpc_xteamr_ui_4x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                            void (*_rf_lds)(_RF_LDS uint32_t *,
                                            _RF_LDS uint32_t *),
                            uint32_t iv)
  { *r_ptr = v;}
void __kmpc_xteamr_l_4x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                           void (*_rf)(long *, long),
                           void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                           long iv)
  { *r_ptr = v;}
void __kmpc_xteamr_ul_4x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                            void (*_rf_lds)(_RF_LDS uint64_t *,
                                            _RF_LDS uint64_t *),
                            uint64_t iv)
  { *r_ptr = v;}
#undef _RF_LDS
#endif
}

// Overloaded functions used in simulated reductions below
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(double val, double* rval,
	       	double* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_d_4x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_d,__kmpc_rfun_sum_lds_d, 0.0);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(float val, float* rval,
	       	float* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_f_4x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_f,__kmpc_rfun_sum_lds_f, 0.0);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(int val, int* rval,
	       	int* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_i_4x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_i,__kmpc_rfun_sum_lds_i, 0);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(unsigned int val, unsigned int* rval,
	       	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ui_4x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_ui,__kmpc_rfun_sum_lds_ui, 0);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(long val, long* rval,
	       	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_l_4x64(val,rval, xteam_mem,td_ptr, __kmpc_rfun_sum_l,__kmpc_rfun_sum_lds_l, 0);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_sum(unsigned long val, unsigned long * rval,
	       	unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ul_4x64(val,rval, xteam_mem,td_ptr, __kmpc_rfun_sum_ul,__kmpc_rfun_sum_lds_ul, 0);
}

#define __XTEAM_MAX_FLOAT (__builtin_inff())
#define __XTEAM_LOW_FLOAT -__XTEAM_MAX_FLOAT
#define __XTEAM_MAX_DOUBLE (__builtin_huge_val())
#define __XTEAM_LOW_DOUBLE -__XTEAM_MAX_DOUBLE
#define __XTEAM_MAX_INT32 2147483647
#define __XTEAM_LOW_INT32 (-__XTEAM_MAX_INT32 - 1)
#define __XTEAM_MAX_UINT32 4294967295
#define __XTEAM_LOW_UINT32 0
#define __XTEAM_MAX_INT64 9223372036854775807
#define __XTEAM_LOW_INT64 (-__XTEAM_MAX_INT64 - 1)
#define __XTEAM_MAX_UINT64 0xffffffffffffffff
#define __XTEAM_LOW_UINT64 0


void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(double val, double* rval,
	double * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_d_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_d,__kmpc_rfun_max_lds_d, __XTEAM_LOW_DOUBLE);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(float val, float* rval,
	float * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_f_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_f,__kmpc_rfun_max_lds_f, __XTEAM_LOW_FLOAT);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(int val, int* rval,
	int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_i_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_i,__kmpc_rfun_max_lds_i, __XTEAM_LOW_INT32);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ui_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_ui,__kmpc_rfun_max_lds_ui, 0u);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(long val, long * rval,
	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_l_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_l,__kmpc_rfun_max_lds_l, __XTEAM_LOW_INT64);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_max(unsigned long val, unsigned long * rval,
	unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ul_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_ul,__kmpc_rfun_max_lds_ul, 0ul);
}

void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(double val, double* rval,
	double * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_d_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_d,__kmpc_rfun_min_lds_d, __XTEAM_MAX_DOUBLE);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(float val, float* rval,
	float * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_f_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_f,__kmpc_rfun_min_lds_f, __XTEAM_MAX_FLOAT);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(int val, int* rval,
	int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_i_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_i,__kmpc_rfun_min_lds_i, __XTEAM_MAX_INT32);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ui_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_ui,__kmpc_rfun_min_lds_ui, __XTEAM_MAX_UINT32);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(long val, long * rval,
	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_l_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_l,__kmpc_rfun_min_lds_l, __XTEAM_MAX_INT64);
}
void __attribute__((flatten, always_inline)) __kmpc_xteamr_min(unsigned long val, unsigned long * rval,
	 unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ul_4x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_ul,__kmpc_rfun_min_lds_ul, __XTEAM_MAX_UINT64);
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
