#include <assert.h>
// Tool related code below
#include <omp-tools.h>

#define OMPT_BUFFER_REQUEST_SIZE 256

// Utilities
static void print_record_ompt(ompt_record_ompt_t *rec) {
#if 0  
  printf("rec=%p type=%d time=%lu thread_id=%lu target_id=%lu\n",
	 rec, rec->type, rec->time, rec->thread_id, rec->target_id);

  switch (rec->type) {
  case ompt_callback_target:
  case ompt_callback_target_emi:
    {
      ompt_record_target_t target_rec = rec->record.target;
      printf("\tTarget task: kind=%d endpoint=%d device=%d task_id=%lu target_id=%lu codeptr=%p\n",
	     target_rec.kind, target_rec.endpoint, target_rec.device_num,
	     target_rec.task_id, target_rec.target_id, target_rec.codeptr_ra);
      break;
    }
  case ompt_callback_target_data_op:
  case ompt_callback_target_data_op_emi:
    {
      ompt_record_target_data_op_t target_data_op_rec = rec->record.target_data_op;
      printf("\tTarget data op: host_op_id=%lu optype=%d src_addr=%p src_device=%d "
	     "dest_addr=%p dest_device=%d bytes=%lu end_time=%lu duration=%luus codeptr=%p\n",
	     target_data_op_rec.host_op_id, target_data_op_rec.optype,
	     target_data_op_rec.src_addr, target_data_op_rec.src_device_num,
	     target_data_op_rec.dest_addr, target_data_op_rec.dest_device_num,
	     target_data_op_rec.bytes, target_data_op_rec.end_time,
	     target_data_op_rec.end_time - rec->time,
	     target_data_op_rec.codeptr_ra);
      break;
    }
  case ompt_callback_target_submit:
  case ompt_callback_target_submit_emi:
    {
      ompt_record_target_kernel_t target_kernel_rec = rec->record.target_kernel;
      printf("\tTarget kernel: host_op_id=%lu requested_num_teams=%u granted_num_teams=%u "
	     "end_time=%lu duration=%luus\n",
	     target_kernel_rec.host_op_id, target_kernel_rec.requested_num_teams,
	     target_kernel_rec.granted_num_teams, target_kernel_rec.end_time,
	     target_kernel_rec.end_time - rec->time);
    break;
    }
  default:
    assert(0);
    break;
  }
#endif  
}

static void delete_buffer_ompt(ompt_buffer_t *buffer) {
  free(buffer);
  printf("Deallocated %p\n", buffer);
}

// OMPT entry point handles
static ompt_set_callback_t ompt_set_callback = 0;
static ompt_start_trace_t ompt_start_trace = 0;
static ompt_flush_trace_t ompt_flush_trace = 0;
static ompt_stop_trace_t ompt_stop_trace = 0;
static ompt_get_record_ompt_t ompt_get_record_ompt = 0;
static ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor = 0;

// OMPT callbacks

// Trace record callbacks
static void on_ompt_callback_buffer_request (
  int device_num,
  ompt_buffer_t **buffer,
  size_t *bytes
) {
  *bytes = OMPT_BUFFER_REQUEST_SIZE;
  *buffer = malloc(*bytes);
  printf("Allocated %lu bytes at %p in buffer request callback\n", *bytes, *buffer);
}

static void on_ompt_callback_buffer_complete (
  int device_num,
  ompt_buffer_t *buffer,
  size_t bytes,
  ompt_buffer_cursor_t begin,
  int buffer_owned
) {
#if 0  
  printf("Executing buffer complete callback: %d %p %lu %p %d\n",
     device_num, buffer, bytes, (void*)begin, buffer_owned);
#endif
  
  ompt_record_ompt_t *rec = ompt_get_record_ompt(buffer, begin);
  while (rec) {
    print_record_ompt(rec);
    ompt_buffer_cursor_t next;
    int status = ompt_advance_buffer_cursor(NULL, /* TODO */
					    buffer,
					    bytes,
					    (ompt_buffer_cursor_t)rec,
					    &next);
    if (!status) break;
    rec = (ompt_record_ompt_t*)next; // call ompt_get_record_ompt
    assert(rec != NULL && "Buffer advanced to nullptr");
  }
  if (buffer_owned) delete_buffer_ompt(buffer);
}

static int start_trace() {
  assert(ompt_start_trace);
  return ompt_start_trace(0, &on_ompt_callback_buffer_request,
			  &on_ompt_callback_buffer_complete);
}

static int flush_trace() {
  assert(ompt_flush_trace);
  return ompt_flush_trace(0);
}

static int stop_trace() {
  assert(ompt_stop_trace);
  return ompt_stop_trace(0);
}

// Synchronous callbacks
static void on_ompt_callback_device_initialize
(
  int device_num,
  const char *type,
  ompt_device_t *device,
  ompt_function_lookup_t lookup,
  const char *documentation
 ) {
#if 0  
  printf("Init: device_num=%d type=%s device=%p lookup=%p doc=%p\n",
	 device_num, type, device, lookup, documentation);
#endif
  
  // TODO
  if (!lookup) {
    printf("Trace collection disabled on device %d\n", device_num);
    return;
  }
  // Add device_num -> device mapping to a map
  
  ompt_start_trace = (ompt_start_trace_t) lookup("ompt_start_trace");
  ompt_flush_trace = (ompt_flush_trace_t) lookup("ompt_flush_trace");
  ompt_stop_trace = (ompt_stop_trace_t) lookup("ompt_stop_trace");
  ompt_get_record_ompt = (ompt_get_record_ompt_t) lookup("ompt_get_record_ompt");
  ompt_advance_buffer_cursor = (ompt_advance_buffer_cursor_t) lookup("ompt_advance_buffer_cursor");
  
  // In many scenarios, this will be a good place to start the
  // trace. If start_trace is called from the main program, the
  // programmer has to be careful to place the call after the first
  // target construct, otherwise the program will fail. This is
  // because this device_init callback is invoked during the first
  // target construct implementation and any start_trace must come
  // afterwards.

  // TODO move the ompt_start_trace to the main program before any
  // target construct and ensure we error out gracefully. The program
  // should not assert or crash.
  start_trace();
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
#if 0  
  printf("Load: device_num:%d filename:%s host_adddr:%p device_addr:%p bytes:%lu\n",
	 device_num, filename, host_addr, device_addr, bytes);
#endif  
}

static void on_ompt_callback_target_data_op
    (
     ompt_scope_endpoint_t endpoint,
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     ompt_target_data_op_t optype,
     void *src_addr,
     int src_device_num,
     void *dest_addr,
     int dest_device_num,
     size_t bytes,
     const void *codeptr_ra
     ) {
#if 0  
  printf("DataOp: endpoint=%d host_op_id=%lu optype=%d src=%p src_device_num=%d "
	 "dest=%p dest_device_num=%d bytes=%lu code=%p\n",
	 endpoint, host_op_id, optype, src_addr, src_device_num,
	 dest_addr, dest_device_num, bytes, codeptr_ra);
#endif  
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
#if 0  
  printf("Target: kind=%d endpoint=%d device_num=%d target_id=%lu code=%p\n",
	 kind, endpoint, device_num, target_id, codeptr_ra);
#endif  
}

static void on_ompt_callback_target_submit
    (
     ompt_scope_endpoint_t endpoint,
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     unsigned int requested_num_teams
     ) {
#if 0  
  printf("TargetSubmit: endpoint=%d target_id=%lu host_op_id=%lu req_num_teams=%d\n",
     endpoint, target_id, host_op_id, requested_num_teams);
#endif  
}

// Init functions
int ompt_initialize(
  ompt_function_lookup_t lookup,
  int initial_device_num,
  ompt_data_t *tool_data)
{
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  
  ompt_set_callback(ompt_callback_device_initialize,
		    (ompt_callback_t)&on_ompt_callback_device_initialize);
  ompt_set_callback(ompt_callback_device_load,
		    (ompt_callback_t)&on_ompt_callback_device_load);
  ompt_set_callback(ompt_callback_target_data_op,
		    (ompt_callback_t)&on_ompt_callback_target_data_op);
  ompt_set_callback(ompt_callback_target,
		    (ompt_callback_t)&on_ompt_callback_target);
  ompt_set_callback(ompt_callback_target_submit,
		    (ompt_callback_t)&on_ompt_callback_target_submit);
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
