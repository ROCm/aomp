#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_image.h"

#include <fcntl.h>
#include <assert.h>
#include "stdio.h"
#include "string.h"

#include <vector>
#include <map>
#include <atomic>
#include <chrono>

#define CHECK(x) do { if((x) != HSA_STATUS_SUCCESS) { assert(false); abort(); } } while(false);


using namespace std;

struct timer {
  const char *func;
  using clock_ty = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock_ty> start, next;

  explicit timer(const char *func): func(func) {
    start = clock_ty::now();
    next = start;
  }

  void checkpoint(const char *func) {
    auto end = clock_ty::now();

    uint64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - next)
            .count();

    printf("%35s: %lu ns (%f ms)\n", func, t, t / 1e6);
    next = clock_ty::now();
  }

  ~timer() {
    auto end = clock_ty::now();

    uint64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    printf("%35s: %lu ns (%f ms)\n", func, t, t / 1e6);
  }
};



struct Device {
  struct Memory {
    hsa_amd_memory_pool_t pool;
    bool fine;
    bool coarse;
    size_t size;
    size_t granule;
  };

  hsa_agent_t agent;
  std::vector<Memory> pools;
};

struct Kernel {
  uint64_t handle;
  uint32_t scratch;
  uint32_t group;
  uint32_t kernarg_size;
  uint32_t kernarg_align;
};

// Assumes bitfield layout is little endian.
// Assumes std::atomic<uint16_t> is binary compatible with uint16_t and uses HW atomics.
union AqlHeader {
  struct {
    uint16_t type     : 8;
    uint16_t barrier  : 1;
    uint16_t acquire  : 2;
    uint16_t release  : 2;
    uint16_t reserved : 3;
  };
  uint16_t raw;
};

union Aql {
  AqlHeader header;
  hsa_kernel_dispatch_packet_t dispatch;
  hsa_barrier_and_packet_t barrier_and;
  hsa_barrier_or_packet_t barrier_or;
};

struct OCLHiddenArgs {
  uint64_t offset_x;
  uint64_t offset_y;
  uint64_t offset_z;
  void* printf_buffer;
  void* enqueue;
  void* enqueue2;
  void* multi_grid;
};

std::vector<Device> cpu, gpu;
Device::Memory host_fine;
Device::Memory dev_mem;

hsa_file_t file;
hsa_code_object_reader_t code_obj_rdr;
hsa_executable_t executable;

bool DeviceDiscovery() {
  hsa_status_t err;

  err = hsa_iterate_agents([](hsa_agent_t agent, void*) {
    hsa_status_t err;

    Device dev;
    dev.agent = agent;

    hsa_device_type_t type;
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    CHECK(err);

    err = hsa_amd_agent_iterate_memory_pools(agent, [](hsa_amd_memory_pool_t pool, void* data) {
      std::vector<Device::Memory>& pools = *reinterpret_cast<std::vector<Device::Memory>*>(data);
      hsa_status_t err;

      hsa_amd_segment_t segment;
      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
      CHECK(err);

      if(segment != HSA_AMD_SEGMENT_GLOBAL)
        return HSA_STATUS_SUCCESS;

      uint32_t flags;
      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
      CHECK(err);

      Device::Memory mem;
      mem.pool=pool;
      mem.fine = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
      mem.coarse = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);

      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &mem.size);
      CHECK(err);

      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &mem.granule);
      CHECK(err);

      pools.push_back(mem);
      return HSA_STATUS_SUCCESS;
    }, (void*)&dev.pools);

    if(!dev.pools.empty()) {
      if(type == HSA_DEVICE_TYPE_CPU)
        cpu.push_back(dev);
      else
        gpu.push_back(dev);
    }

    return HSA_STATUS_SUCCESS;
  }, nullptr);

  []() {
    for(auto& dev : cpu) {
      for(auto& mem : dev.pools) {
        if(mem.fine) {
          host_fine = mem;
          return;
        }
      }
    }
  }();

  []() {
    for(auto& dev : gpu) {
      for(auto& mem : dev.pools) {
	if(mem.coarse) {
          dev_mem = mem;
          return;
	}
      }
    }
  }();

  if(cpu.empty() || gpu.empty() || host_fine.pool.handle == 0)
    return false;
  return true;
}

bool LoadCodeObject(std::string filename, hsa_agent_t agent) {
  hsa_status_t err;

  file = open(filename.c_str(), O_RDONLY);
  if(file == -1)
    return false;

  err = hsa_code_object_reader_create_from_file(file, &code_obj_rdr);
  CHECK(err);

  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &executable);
  CHECK(err);

  err = hsa_executable_load_agent_code_object(executable, agent, code_obj_rdr, nullptr, nullptr);
  if(err != HSA_STATUS_SUCCESS)
    return false;

  err = hsa_executable_freeze(executable, nullptr);
  CHECK(err);

  return true;
}

bool GetKernel(std::string kernel, hsa_agent_t agent, Kernel &kern) {
  hsa_executable_symbol_t symbol;
  hsa_status_t err = hsa_executable_get_symbol_by_name(executable, kernel.c_str(), &agent, &symbol);
  if(err != HSA_STATUS_SUCCESS) {
    err = hsa_executable_get_symbol_by_name(executable, (kernel+".kd").c_str(), &agent, &symbol);
    if(err != HSA_STATUS_SUCCESS) {
      return false;
    }
  }

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kern.handle);
  CHECK(err);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kern.scratch);
  CHECK(err);
  //printf("Scratch: %d\n", kern.scratch);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kern.group);
  CHECK(err);
  //printf("LDS: %d\n", kern.group);

  // Remaining needs code object v2 or comgr.
  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kern.kernarg_size);
  CHECK(err);
  //printf("Kernarg Size: %d\n", kern.kernarg_size);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &kern.kernarg_align);
  CHECK(err);
  //printf("Kernarg Align: %d\n", kern.kernarg_align);

  return true;
}

// Not for parallel insertion.
bool SubmitPacket(hsa_queue_t* queue, Aql& pkt) {
  size_t mask = queue->size - 1;
  Aql* ring = (Aql*)queue->base_address;

  uint64_t write = hsa_queue_load_write_index_relaxed(queue);
  uint64_t read = hsa_queue_load_read_index_relaxed(queue);
  if(write - read + 1 > queue->size)
    return false;

  Aql& dst = ring[write & mask];

  uint16_t header = pkt.header.raw;
  pkt.header.raw = dst.header.raw;
  dst = pkt;
  __atomic_store_n(&dst.header.raw, header, __ATOMIC_RELEASE);
  pkt.header.raw = header;

  hsa_queue_store_write_index_release(queue, write+1);
  hsa_signal_store_screlease(queue->doorbell_signal, write);

  return true;
}

#define N 30000

int main(int argc, char* argv[]) {
  int device_index=0;
  bool coarse = false;
  if(argc==2)
    coarse = argv[1][0] == '1';

  timer stopwatch("main()");

  hsa_status_t err;
  err = hsa_init();
  CHECK(err);

  if(!DeviceDiscovery()) {
    printf("Usable devices not found.\n");
    return 0;
  }

  if(!LoadCodeObject("large_array_copy.hsaco", gpu[device_index].agent)) {
    printf("Kernel file not found or not usable with given agent.\n");
    return 0;
  }

  Kernel test;
  if(!GetKernel("large_array_copy", gpu[device_index].agent, test)) {
    printf("Test kernel not found.\n");
    return 0;
  }

  hsa_queue_t* queue;
  err = hsa_queue_create(gpu[device_index].agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &queue);
  CHECK(err);

  struct args_t {
    int* a;
    int* b;
    OCLHiddenArgs hidden;
  };

  args_t* args;
  size_t size = ((sizeof(args_t) + host_fine.granule - 1) / host_fine.granule) * host_fine.granule;
  err = hsa_amd_memory_pool_allocate(host_fine.pool, size, 0, (void**)&args);
  CHECK(err);

  err = hsa_amd_agents_allow_access(1, &gpu[device_index].agent, nullptr, args);
  CHECK(err);

  memset(args, 0, size);

  hsa_signal_t signal;
  // Use interrupts.
  err = hsa_amd_signal_create(1, 0, nullptr, 0, &signal);
  CHECK(err);

  Aql packet = {0};
  packet.header.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  packet.header.barrier = 1;
  packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
  packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

  packet.dispatch.setup = 1;
  packet.dispatch.workgroup_size_x = 256;
  packet.dispatch.workgroup_size_y = 1;
  packet.dispatch.workgroup_size_z = 1;
  packet.dispatch.grid_size_x = N*N;
  packet.dispatch.grid_size_y = 1;
  packet.dispatch.grid_size_z = 1;

  packet.dispatch.group_segment_size = test.group;
  packet.dispatch.private_segment_size = test.scratch;
  packet.dispatch.kernel_object = test.handle;

  packet.dispatch.kernarg_address = args;
  packet.dispatch.completion_signal = signal;


  // End boilerplate - start ported omp test
  int n = N*N;
  int *a = new int[n];
  int *b = new int[n];
  int *d_a, *d_b;
  for(int i = 0; i < n; i++)
      b[i] = i;

  //#pragma omp target parallel for map(to:b[:n])
  //for(int i = 0; i < n; i++)
  //    a[i] = b[i];

  // allocate a and b on device
  err = hsa_amd_memory_pool_allocate(dev_mem.pool, n*sizeof(int), 0, (void **)&d_a);
  if (err != HSA_STATUS_SUCCESS) {
    printf("Error allocating a\n");
    return err;
  }

  err = hsa_amd_memory_pool_allocate(dev_mem.pool, n*sizeof(int), 0, (void **)&d_b);
  if (err != HSA_STATUS_SUCCESS) {
    printf("Error allocating b\n");
    return err;
  }

  err = hsa_amd_agents_allow_access(1, &gpu[device_index].agent, nullptr, d_a);
  CHECK(err);

  err = hsa_amd_agents_allow_access(1, &gpu[device_index].agent, nullptr, d_b);
  CHECK(err);


  stopwatch.checkpoint("Init");

  // copy b from host to device
  hsa_status_t rc = hsa_memory_copy(d_b, b, n*sizeof(int));
  printf("Size = %zu\n", n*sizeof(int));
  if (rc != HSA_STATUS_SUCCESS) {
    printf("Error copy from host to device\n");
    return -1;
  }

  stopwatch.checkpoint("H2D");


  args->a = d_a;
  args->b = d_b;
  SubmitPacket(queue, packet);
  hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, -1, HSA_WAIT_STATE_BLOCKED);

  stopwatch.checkpoint("Kernel");

  // copy a from device to host
  rc = hsa_memory_copy(a, d_a, n*sizeof(int));
  if (rc != HSA_STATUS_SUCCESS) {
    printf("Error copy from device to host\n");
    return -1;
  }

  stopwatch.checkpoint("D2H");

  // for(int i = 0; i < 10; i++)
  //   printf("a[%d] = %d\n", i, a[i]);
  // return 0;

  int num_err = 0;
  for(int i = 0; i < n; i++) {
    if(a[i] != i) {
      printf("error at %d: expected %d, got %d\n", i, i, a[i]);
      if(++num_err > 10) break;
    }
  }

  //SPK: Added so that success shows something different from a silent crash.
  printf("Done\n");

  stopwatch.checkpoint("End");

  return 0;
}
