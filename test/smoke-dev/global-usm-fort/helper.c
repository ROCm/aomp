#include "ISO_Fortran_binding.h"
#include <stdio.h>

void print_descr(const void *name_, const void *p)
{
  // Fortran string is string pointer followed by string length
  const char *str = ((const char **)name_)[0];
  int len = ((const long *)name_)[1];
  char name[len+1];
  name[len] = 0;
  for (int i = 0; i < len; ++i)
    name[i] = str[i];

  // access fields of descriptor
  const CFI_cdesc_t *descr = p;

  // Are we on host or target?
#if defined(__AMDGCN__)
# define SOURCE "TARGET"
#else
# define SOURCE "  HOST"
#endif

  if (descr->version != CFI_VERSION)
    printf("WARNING: descr->version is %d should be %d\n",
      descr->version, CFI_VERSION);
  printf("=== %s: name=%s descr=%15p base=%15p [sz:%lu,el:%lu,r:%d,t:%d,a:%d]\n",
    SOURCE, name, descr, descr->base_addr,
    sizeof(CFI_cdesc_t) + 3*sizeof(CFI_index_t)*descr->rank,
    descr->elem_len, descr->rank, descr->type, descr->attribute);
}
#pragma omp declare target (print_descr)

void print_descr_real (const void *name_, const void *p) __attribute__((alias("print_descr")));
void print_descr_double(const void *name_, const void *p) __attribute__((alias("print_descr")));

