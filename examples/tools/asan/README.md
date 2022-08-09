The AOMP compiler supports ASan for AMDGPU with applications written in both HIP and OpenMP.
========================================================================================================================================

Address Sanitizer(ASan) is a memory error detector tool utilized by applications to detect various errors ranging from spatial issues like out-of-bound access to temporal issues like use-after-free.

* **Features Supported On Host Platfrom(Target x86_64).**
   1. Use after free
   2. Buffer Overflows
        - Heap buffer overflow
        - Stack buffer overflow
        - Global buffer overflow
   3. Use after return
   4. Use after scope
   5. Initialization order bugs

 * **Features Supported on AMDGPU Platform(amdgcn-amd-amdhsa)**
   1. Heap buffer overflow
   2. Global buffer overflow

Requirements
========================================================================================================================================

* **Software(Kernel/OS) Requirements**

    - **Unified Memory Support(HMM Kernel)**
        1. This feature requires Linux Kernel versions greater than 5.14.
        2. This feature also requires latest KFD driver packaged in ROCm stack.
        3. Unified memory support can be tested with applications compiled with xnack capability.

    - **XNACK Capability**
        1. Xnack replay enabled mode compiled binaries on execution indicates that runtime can handle page faults gracefully.So,if any page faults occur on gpu then at runtime a retry of that memory access happens.
              > xnack+ --offload-arch=gfx908:xnack+
        2. Xnack replay disabled mode compiled binaries on indicates that runtime can't handle page faults on GPU.So, developer should write offloading kernels with caution that it should not create any page faults on GPU.
              > xnack- with –offload-arch=gfx908:xnack-
