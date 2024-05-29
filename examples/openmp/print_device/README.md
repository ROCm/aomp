
Prints the value of omp_get_num_devices() and the contents of ROCR_VISIBLE_DEVICES environment variable.

The "run" target will run print_device 3 times 
 - without ROCR_VISIBLE_DEVICES set, 
 - with it set to 0,1
 - and with the gpurun wrapper utility.

The first execution will show you the number of all the GPUs on the system. The 2nd execution will 
show two GPUs if there are two, and the third will only show one because the default mode for gpurun is to 
use a single GPU. Notice that GPURUN will set a value for ROCR_VISIBLE_DEVICES. 

There is another target called "runmpi".  If mpirun is installed, this will show that gpurun will automatically assign a different GPU to each process if multiple GPUs are available. 
