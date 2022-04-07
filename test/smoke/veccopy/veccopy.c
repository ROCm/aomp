#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 10;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong varlue: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

// Tool related code below
#include <omp-tools.h>

// OMPT entry point handles
static ompt_set_callback_t ompt_set_callback;

// OMPT callbacks

// Synchronous callbacks
static void on_ompt_callback_device_initialize
(
  int device_num,
  const char *type,
  ompt_device_t *device,
  ompt_function_lookup_t lookup,
  const char *documentation
 ) {
  printf("Initialize: device_num=%d type=%s device=%p lookup=%p doc=%p\n",
	 device_num, type, device, lookup, documentation);
}

static void on_ompt_callback_device_load
    (
     int device_num,
     const char *filename,
     int64_t offset_in_file,
     void *vma_in_file,
     size_t bytes,
     void *host_addr,
     void *device_addr,
     uint64_t module_id
     ) {
  printf("Load: device_num=%d file=%s host=%p device_addr=%p bytes=%lu\n",
     device_num, filename, host_addr, device_addr, bytes);
}

static void on_ompt_callback_target
    (
     ompt_target_t kind,
     ompt_scope_endpoint_t endpoint,
     int device_num,
     ompt_data_t *task_data,
     ompt_id_t target_id,
     const void *codeptr_ra
     ) {
  printf("Target: kind=%d endpoint=%d device_num=%d target_id=%lu code=%p\n",
	 kind, endpoint, device_num, target_id, codeptr_ra);
}

// Init functions
int ompt_initialize(
  ompt_function_lookup_t lookup,
  int initial_device_num,
  ompt_data_t *tool_data)
{
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");

  if (ompt_set_callback) {
    ompt_set_callback(ompt_callback_device_initialize,
		      (ompt_callback_t)&on_ompt_callback_device_initialize);
    ompt_set_callback(ompt_callback_device_load,
		      (ompt_callback_t)&on_ompt_callback_device_load);
    ompt_set_callback(ompt_callback_target,
		      (ompt_callback_t)&on_ompt_callback_target);
  }
  return 1; //success
}

void ompt_finalize(ompt_data_t *tool_data)
{
}

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#ifdef __cplusplus
}
#endif

