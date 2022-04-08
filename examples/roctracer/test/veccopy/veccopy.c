#include <stdio.h>
#include <atomic>
#include <omp.h>

#define AMD_INTERNAL_BUILD

#include <ext/hsa_rt_utils.hpp>
#include "hsa.h"
#include "src/core/trace_buffer.h"

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>

#ifdef __cplusplus
static thread_local const size_t msg_size = 512;
static thread_local char* msg_buf = NULL;
static thread_local char* message = NULL;
#else
static const size_t msg_size = 512;
static char* msg_buf = NULL;
static char* message = NULL;
#endif
void SPRINT(const char* fmt, ...) {
  if (msg_buf == NULL) {
    msg_buf = (char*) calloc(msg_size, 1);
    message = msg_buf;
  }

  va_list args;
  va_start(args, fmt);
  message += vsnprintf(message, msg_size - (message - msg_buf), fmt, args);
  va_end(args);
}
void SFLUSH() {
  if (msg_buf == NULL) abort();
  message = msg_buf;
  msg_buf[msg_size - 1] = 0;
  fprintf(stdout, "%s", msg_buf);
  fflush(stdout);
}

#ifndef ITERATIONS
# define ITERATIONS 101
#endif

int iterations = ITERATIONS;
void init_tracing();
void start_tracing();
void stop_tracing();

int main()
{
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++){
    a[i]=0;
    b[i]=i;
  }

  init_tracing();

  roctracer_start();

#pragma omp target parallel for map(from: a[0:N]) map(to: b[0:N])
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  roctracer_stop();

  stop_tracing();

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc){
    printf("Success\n");
    return EXIT_SUCCESS;
  } else{
    printf("Failure\n");
    return EXIT_FAILURE;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
//
#if 1
#include <roctracer_hcc.h>
#include <roctracer_hsa.h>
#include <roctracer_kfd.h>
#include <roctracer_roctx.h>

#include <unistd.h> 
#include <sys/syscall.h>   /* For SYS_xxx definitions */ 

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      fprintf(stderr, "%s\n", roctracer_error_string());                                                    \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

static inline uint32_t GetTid() { return syscall(__NR_gettid); }
static inline uint32_t GetPid() { return syscall(__NR_getpid); }

typedef hsa_rt_utils::Timer::timestamp_t timestamp_t;
hsa_rt_utils::Timer* timer = NULL;
thread_local timestamp_t hsa_begin_timestamp = 0;
thread_local timestamp_t hip_begin_timestamp = 0;
thread_local timestamp_t kfd_begin_timestamp = 0;

struct hsa_api_trace_entry_t {
  std::atomic<uint32_t> valid;
  roctracer::entry_type_t type;
  uint32_t cid;
  timestamp_t begin;
  timestamp_t end;
  uint32_t pid;
  uint32_t tid;
  hsa_api_data_t data;
};

void hsa_api_flush_cb(hsa_api_trace_entry_t* entry);
constexpr roctracer::TraceBuffer<hsa_api_trace_entry_t>::flush_prm_t hsa_flush_prm = {roctracer::DFLT_ENTRY_TYPE, hsa_api_flush_cb};
roctracer::TraceBuffer<hsa_api_trace_entry_t>* hsa_api_trace_buffer = NULL;

TRACE_BUFFER_INSTANTIATE();

// Runtime API callback function
void api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;

  if (domain == ACTIVITY_DOMAIN_ROCTX) {
    const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
    fprintf(stdout, "rocTX <\"%s pid(%d) tid(%d)\">\n", data->args.message, GetPid(), GetTid());
    return;
  }
  if (domain == ACTIVITY_DOMAIN_KFD_API) {
    const kfd_api_data_t* data = (const kfd_api_data_t*)(callback_data);
    fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)>\n",
        roctracer_op_string(ACTIVITY_DOMAIN_KFD_API, cid, 0),
        cid,
        data->correlation_id,
        (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
    return;
  }
#if 0
  const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
  SPRINT("<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)> ",
    roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0),
    cid,
    data->correlation_id,
    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        SPRINT("dst(%p) src(%p) size(0x%x) kind(%u)",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        SPRINT("ptr(%p) size(0x%x)",
          data->args.hipMalloc.ptr,
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        SPRINT("ptr(%p)", data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        SPRINT("kernel(\"%s\") stream(%p)",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        SPRINT("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
  }
  SPRINT("\n");
  SFLUSH();
#endif
}
// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  SPRINT("\tActivity records:\n");
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    SPRINT("\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
      name,
      record->correlation_id,
      record->begin_ns,
      record->end_ns);
    if ((record->domain == ACTIVITY_DOMAIN_HIP_API) || (record->domain == ACTIVITY_DOMAIN_KFD_API)) {
      SPRINT(" process_id(%u) thread_id(%u)",
        record->process_id,
        record->thread_id);
    } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      SPRINT(" device_id(%d) queue_id(%lu)",
        record->device_id,
        record->queue_id);
      if (record->op == HIP_OP_ID_COPY) SPRINT(" bytes(0x%zx)", record->bytes);
    } else if (record->domain == ACTIVITY_DOMAIN_HSA_OPS) {
      SPRINT(" se(%u) cycle(%lu) pc(%lx)",
        record->pc_sample.se,
        record->pc_sample.cycle,
        record->pc_sample.pc);
    } else if (record->domain == ACTIVITY_DOMAIN_EXT_API) {
      SPRINT(" external_id(%lu)", record->external_id);
    } else {
      fprintf(stderr, "Bad domain %d\n\n", record->domain);
      abort();
    }
    SPRINT("\n");
    SFLUSH();

    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}

void hsa_api_flush_cb(hsa_api_trace_entry_t* entry) {
#if 0  
  std::ostringstream os;
  os << entry->begin << ":" << entry->end << " " << entry->pid << ":" << entry->tid << " " << hsa_api_data_pair_t(entry->cid, entry->data);
  fprintf(std::cout, "%s\n", os.str().c_str()); fflush(std::cout);
#endif  
}

// HSA API callback function
void hsa_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    hsa_begin_timestamp = timer->timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down) ? hsa_begin_timestamp : timer->timestamp_fn_ns();
    hsa_api_trace_entry_t* entry = hsa_api_trace_buffer->GetEntry();
    entry->cid = cid;
    entry->begin = hsa_begin_timestamp;
    entry->end = end_timestamp;
    entry->pid = GetPid();
    entry->tid = GetTid();
    entry->data = *data;
    entry->valid.store(roctracer::TRACE_ENTRY_COMPL, std::memory_order_release);
  }
}

// Init tracing routine
void init_tracing() {
  printf("# INIT #############################\n");
  // roctracer properties
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
  // Allocating tracing pool
  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  ROCTRACER_CALL(roctracer_open_pool(&properties));

  hsa_api_trace_buffer = new roctracer::TraceBuffer<hsa_api_trace_entry_t>("HSA API", 0x200000, &hsa_flush_prm, 1);
  
  // Enable HIP API callbacks
  // ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL));
  // Enable HIP activity tracing
#if HIP_API_ACTIVITY_ON
  ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
#endif
  ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
  // Enable HSA domain tracing
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, NULL));
  // Enable PC sampling
  ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
  // Enable KFD API tracing
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_KFD_API, api_callback, NULL));
  // Enable rocTX
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL));
}
#if 0
// Start tracing routine
void start_tracing() {
  printf("# START (%d) #############################\n", iterations);
  // Start
  if ((iterations & 1) == 1) roctracer_start();
  else roctracer_stop();
}
#endif
// Stop tracing routine
void stop_tracing() {
  ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
#if HIP_API_ACTIVITY_ON
  ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
#endif
  ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
  ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_API));
  ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
  ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API));
  ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
  ROCTRACER_CALL(roctracer_flush_activity());
  printf("# STOP  #############################\n");
}
#else
void init_tracing() {}
void start_tracing() {}
void stop_tracing() {}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
