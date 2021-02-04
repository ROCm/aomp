#include <omp.h>
#include <stdio.h>
#include <sched.h>

int main( int argc, char **argv){

        int execution_space_gpu = 0;
        if (argc > 1){
		fprintf(stderr,"argv[0] = %s\n",argv[1]);
                execution_space_gpu = atoi(argv[1]);
	}

	int N = 1024*1024*100;
	int Niter = 10;
	float *A, *B;
	A = new float[N];
	B = new float[N];
        double GB = (double)N * sizeof(float) / (1024.0*1024.0*1024.0);

	int ndevices = omp_get_device_num();
	printf("ndevices= %d\n",ndevices);

	int cpuid[omp_get_max_threads()];
        #pragma omp parallel
	{
          cpuid[ omp_get_thread_num()] = sched_getcpu();
	}
        for (int i=0; i < omp_get_max_threads(); ++i)
		printf("tid = %d, cpuid = %d\n",i,cpuid[i]);

        #pragma omp parallel for
        for (int i=0; i < N; ++i){
		A[i] = i*0.0001;
		B[i] = 0.0;
	}

        #pragma omp target enter data map(to:A[0:N],B[0:N]) if(execution_space_gpu)

        #pragma omp target teams distribute parallel for thread_limit(512) num_teams(120*10) schedule(static,1) if(target:execution_space_gpu)
        for (int i=0; i < N; ++i) B[i] = omp_get_thread_num();

#if 1
        #pragma omp target update from(B[0:N]) if(execution_space_gpu)

        for (int i=0; i < 70*1; i+=1)
           printf(" B[%d] = %g\n",i,B[i]);
#endif

	double t1 = omp_get_wtime();
	for (int iter = 0 ; iter < Niter; ++iter){
        #pragma omp target teams distribute parallel for if(target:execution_space_gpu) 
    	  for (int i=0; i < N; ++i) B[i] = A[i];
	}
        double t2 = omp_get_wtime();
        printf("memcpy time = %g [s]   BW = %g [GB/s]\n",(t2-t1)/Niter, 2.0*GB/((t2-t1)/Niter));

        #pragma omp target exit data map(release:A[0:N]) map(from:B[0:N]) if(execution_space_gpu)


        for (int i=0; i < 10; ++i)
	   printf("A[%d] = %g, B[%d] = %g\n",i,A[i],i,B[i]);


        delete[] A;
	delete[] B;

	return 0;

}

