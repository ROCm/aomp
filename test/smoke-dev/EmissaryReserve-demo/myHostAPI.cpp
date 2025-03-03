#include "omp.h"
#include <stdio.h>
#include <stdarg.h>
#include <EmissaryIds.h>

///  STEP 0 , You have some host-only API you want to make available from 
///           an offload device.  These are the two sample functions foo and bar. 
///           These are typically in a well-tested host library.

///  STEP 1, Create an enum with an entry for each targetted function 
///          to become part of the RESERVE Emissary API. 

typedef enum {
  _foo_idx,
  _bar_idx
} my_api_fns;


///  STEP 2  Create device variants with same interface that simply call
///          _emissary_exec with an new first argument that identifies
///          the function to execute on the host.

#pragma omp declare target
extern "C" int foo(int *buf, int count, int tag ) ;
extern "C" int bar(int *buf, int count, int source, int tag) ;
#pragma omp begin declare variant match(device={kind(nohost)})
extern "C" int foo(int *buf, int count, int tag ) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_RESERVE, _foo_idx),
    buf, count, tag );
}
extern "C" int bar(int *buf, int count, int source, int tag) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_RESERVE, _bar_idx),
    buf, count, source, tag);
}
#pragma omp end declare variant
#pragma omp end declare target

///  These are the host definitions of your API functions
extern "C" int foo(int *array, int count, int tag ) {
  int ary_value = (int) array[0];
  printf("INSIDE HOST API FUNCTION foo &array:%p ary_value:%d count:%d tag:%d \n",
    array, ary_value, count, tag) ;
  return ary_value+count+tag;
}
extern "C" int bar(int *array, int count, int source, int tag) {
  int ary_value = (int) array[1];
  printf("INSIDE HOST API FUNCTION bar &array:%p ary_value:%d count:%d tag:%d source:%d\n",
    array, ary_value, count, tag, source);
  return ary_value+count+tag+source;
}

///   STEP 3 Create a variadic wrapper functions for each function.
///          This function assembles the arguments and then calls the
///          actual API. It may take appropriate action depending on 
///          argument types and values. 

int V_foo(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *buf  = va_arg(args, int *);
  int count  = va_arg(args, int);
  int tag    = va_arg(args, int);
  va_end(args);
  return foo(buf,count,tag);
}
int V_bar(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *buf = va_arg(args, int *);
  int count  = va_arg(args, int);
  int source = va_arg(args, int);
  int tag    = va_arg(args, int);
  va_end(args);
  return bar(buf,count,source,tag);
}

///   STEP 4 Create the host function EmissaryReserve that the rpc thread 
///          calls via Emissary when EMIS_ID_RESERVE is encountered.
///          The helper function EmissaryBuildVargs prepares the args for 
///          the call to the variadic wrapper functions.
///          FIXME: Move EmissaryBuildVargs into the Emissary function

extern "C" emis_return_t EmissaryReserve(char *data, emisArgBuf_t *ab,
		emis_argptr_t * a[MAXVARGS]) {
  switch (ab->emisfnid) {
  case _foo_idx: {
    void *fnptr = (void *)V_foo;
    int return_value_int =
      V_foo(fnptr, a[0], a[1], a[2]);
    return (emis_return_t) return_value_int;
  }
  case _bar_idx: {
    void *fnptr = (void *)V_bar;
    int return_value_int =
      V_bar(fnptr, a[0], a[1], a[2], a[3]);
    return (emis_return_t) return_value_int;
  }
  }
  return (emis_return_t)0;
}
