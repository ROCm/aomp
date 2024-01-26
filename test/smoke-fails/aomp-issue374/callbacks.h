#include <assert.h>
#include <memory>
#include <unordered_set>

// Tool related code below
#include <omp-tools.h>

#define OMPT_BUFFER_REQUEST_SIZE 256

// Map of devices traced
typedef std::unordered_set<ompt_device_t*> DeviceMap_t;
typedef std::unique_ptr<DeviceMap_t> DeviceMapPtr_t;
extern DeviceMapPtr_t DeviceMapPtr;

// Utilities
static void print_record_ompt(ompt_record_ompt_t *rec) {
  if (rec == NULL) return;
  
  printf("rec=%p type=%d time=%lu thread_id=%lu target_id=%lu\n",
	 rec, rec->type, rec->time, rec->thread_id, rec->target_id);

  switch (rec->type) {
  case ompt_callback_target:
  case ompt_callback_target_emi:
    {
      ompt_record_target_t target_rec = rec->record.target;
      printf("\tRecord Target: kind=%d endpoint=%d device=%d task_id=%lu target_id=%lu codeptr=%p\n",
	     target_rec.kind, target_rec.endpoint, target_rec.device_num,
	     target_rec.task_id, target_rec.target_id, target_rec.codeptr_ra);
      break;
    }
  case ompt_callback_target_data_op:
  case ompt_callback_target_data_op_emi:
    {
      ompt_record_target_data_op_t target_data_op_rec = rec->record.target_data_op;
      printf("\t  Record DataOp: host_op_id=%lu optype=%d src_addr=%p src_device=%d "
	     "dest_addr=%p dest_device=%d bytes=%lu end_time=%lu duration=%lu ns codeptr=%p\n",
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
      printf("\t  Record Submit: host_op_id=%lu requested_num_teams=%u granted_num_teams=%u "
	     "end_time=%lu duration=%lu ns\n",
	     target_kernel_rec.host_op_id, target_kernel_rec.requested_num_teams,
	     target_kernel_rec.granted_num_teams, target_kernel_rec.end_time,
	     target_kernel_rec.end_time - rec->time);
    break;
    }
  default:
    assert(0);
    break;
  }
}

static void delete_buffer_ompt(ompt_buffer_t *buffer) {
  free(buffer);
  printf("Deallocated %p\n", buffer);
}

// OMPT entry point handles
static ompt_set_callback_t ompt_set_callback = 0;
static ompt_set_trace_ompt_t ompt_set_trace_ompt = 0;
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

// Note: This callback must handle a null begin cursor. Currently,
// ompt_get_record_ompt, print_record_ompt, and
// ompt_advance_buffer_cursor handle a null cursor.
static void on_ompt_callback_buffer_complete (
  int device_num,
  ompt_buffer_t *buffer,
  size_t bytes, /* bytes returned in this callback */
  ompt_buffer_cursor_t begin,
  int buffer_owned
) {
  printf("Executing buffer complete callback: %d %p %lu %p %d\n",
	 device_num, buffer, bytes, (void*)begin, buffer_owned);

  int status = 1;
  ompt_buffer_cursor_t current = begin;
  while (status) {
    ompt_record_ompt_t *rec = ompt_get_record_ompt(buffer, current);
    print_record_ompt(rec);
    status = ompt_advance_buffer_cursor(NULL, /* TODO device */
					buffer,
					bytes,
					current,
					&current);
  }
  if (buffer_owned) delete_buffer_ompt(buffer);
}

static ompt_set_result_t set_trace_ompt() {
  if (!ompt_set_trace_ompt) return ompt_set_error;

  ompt_set_trace_ompt(0, 1, ompt_callback_target);
  ompt_set_trace_ompt(0, 1, ompt_callback_target_data_op);
  ompt_set_trace_ompt(0, 1, ompt_callback_target_submit);

  return ompt_set_always;
}

static int start_trace(int device_num, ompt_device_t *Device) {
  if (!ompt_start_trace) return 0;

  // This device will be traced.
  assert(DeviceMapPtr->find(Device) == DeviceMapPtr->end() &&
	 "Device already present in the map");
  DeviceMapPtr->insert(Device);

  return ompt_start_trace(Device, &on_ompt_callback_buffer_request,
			  &on_ompt_callback_buffer_complete);
}

static int flush_trace(ompt_device_t *Device) {
  if (!ompt_flush_trace) return 0;
  return ompt_flush_trace(Device);
}

static int stop_trace(ompt_device_t *Device) {
  if (!ompt_stop_trace) return 0;
  return ompt_stop_trace(Device);
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
  printf("Init: device_num=%d type=%s device=%p lookup=%p doc=%p\n",
	 device_num, type, device, lookup, documentation);
  if (!lookup) {
    printf("Trace collection disabled on device %d\n", device_num);
    return;
  }

  ompt_set_trace_ompt = (ompt_set_trace_ompt_t) lookup("ompt_set_trace_ompt");
  ompt_start_trace = (ompt_start_trace_t) lookup("ompt_start_trace");
  ompt_flush_trace = (ompt_flush_trace_t) lookup("ompt_flush_trace");
  ompt_stop_trace = (ompt_stop_trace_t) lookup("ompt_stop_trace");
  ompt_get_record_ompt = (ompt_get_record_ompt_t) lookup("ompt_get_record_ompt");
  ompt_advance_buffer_cursor = (ompt_advance_buffer_cursor_t) lookup("ompt_advance_buffer_cursor");

  // DeviceMap must be initialized only once. Ensure this logic does not
  // depend on external data structures because this init function may be
  // called before main.
  static bool IsDeviceMapInitialized = false;
  if (!IsDeviceMapInitialized) {
    DeviceMapPtr = std::make_unique<DeviceMap_t>();
    IsDeviceMapInitialized = true;
  }

  set_trace_ompt();

  // In many scenarios, this will be a good place to start the
  // trace. If start_trace is called from the main program before this
  // callback is dispatched, the start_trace handle will be null. This
  // is because this device_init callback is invoked during the first
  // target construct implementation.

  start_trace(device_num, device);
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
  printf("Load: device_num:%d filename:%s host_adddr:%p device_addr:%p bytes:%lu\n",
	 device_num, filename, host_addr, device_addr, bytes);
}

static void on_ompt_callback_target_data_op
    (
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
  printf("  Callback DataOp: target_id=%lu host_op_id=%lu optype=%d src=%p src_device_num=%d "
	 "dest=%p dest_device_num=%d bytes=%lu code=%p\n",
	 target_id, host_op_id, optype, src_addr, src_device_num,
	 dest_addr, dest_device_num, bytes, codeptr_ra);
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
  printf("Callback Target: target_id=%lu kind=%d endpoint=%d device_num=%d code=%p\n",
	 target_id, kind, endpoint, device_num, codeptr_ra);
}

static void on_ompt_callback_target_submit
    (
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     unsigned int requested_num_teams
     ) {
  printf("  Callback Submit: target_id=%lu host_op_id=%lu req_num_teams=%d\n",
     target_id, host_op_id, requested_num_teams);
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
