#include <stdio.h>
#include <omp.h>
#include "../utilities/check.h"

#define Arr(x,i,j,k) ((x)[(i)*(I)*(I) + (j)*(I) + (k)])

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_device_parallel(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - parallel...\n");

  unsigned A[I*I*I];
  unsigned B[I*I*I];
  //unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I]) 
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
  
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp parallel for schedule(static,1) 
    for (unsigned j=0; j<I; ++j)
    {

      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
         
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  //free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_device_parallel_parallel(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - parallel + parallel...\n");

  unsigned A[I*I*I];
  unsigned B[I*I*I];
  //unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp parallel for schedule(static,1) 
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
        
      #pragma omp parallel for schedule(static,1)   
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  //free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_device_simd(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - simd ...\n");

  unsigned A[I*I*I];
  unsigned B[I*I*I];
  //unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp simd 
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
         
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  //free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_device_parallel_simd(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - parallel + simd...\n");

  unsigned A[I*I*I];
  unsigned B[I*I*I];
  //unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
    //printf("B: Inside kernel!\n");
  
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp parallel for 
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
        
      #pragma omp simd  
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  //free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo(unsigned ii) {
  printf("Running host version...\n");

  unsigned A[I*I*I];
  unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  for (unsigned i = 0; i<I; ++i)
  {
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
        
      for (unsigned k = 0; k<I; ++k)
      {
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with host version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (host) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  foo_device_parallel<I,P1,P2>         (ii, &A[0], &B[0]);
  foo_device_parallel_parallel<I,P1,P2>(ii, &A[0], &B[0]);
  foo_device_simd<I,P1,P2>             (ii, &A[0], &B[0]);
  foo_device_parallel_simd<I,P1,P2>    (ii, &A[0], &B[0]);
  free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_ptr_device_parallel(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - parallel...\n");

  unsigned *A = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
  
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp parallel for schedule(static,1) 
    for (unsigned j=0; j<I; ++j)
    {

      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
         
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  free(A);
  free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_ptr_device_parallel_parallel(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - parallel + parallel...\n");

  unsigned *A = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp parallel for schedule(static,1) 
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
        
      #pragma omp parallel for schedule(static,1)   
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  free(A);
  free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_ptr_device_simd(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - simd ...\n");

  unsigned *A = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp simd 
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
         
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  free(A);
  free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_ptr_device_parallel_simd(unsigned ii, unsigned *ReferenceA, unsigned *ReferenceB) {
  printf("Running device version - parallel + simd...\n");

  unsigned *A = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  #pragma omp target map(tofrom:A[:I*I*I],B[:I*I*I])
  #pragma omp teams num_teams(64)
  #pragma omp distribute
  for (unsigned i = 0; i<I; ++i)
  {
    //printf("B: Inside kernel!\n");
  
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    #pragma omp parallel for 
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
        
      #pragma omp simd  
      for (unsigned k = 0; k<I; ++k)
      {       
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with device version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (device) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  if(ReferenceA)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (A[i] != ReferenceA[i]) {
        printf("Error A[i] != ReferenceA[i] at %08x: %08x != %08x\n", i, A[i], ReferenceA[i]);
        break;
      }
    
  if(ReferenceB)
    for(unsigned i = 0; i<I*I*I; ++i)
      if (B[i] != ReferenceB[i]) {
        printf("Error B[i] != ReferenceB[i] at %08x: %08x != %08x\n", i, B[i], ReferenceB[i]);
        break;
      }

  free(A);
  free(B);
}

template<const unsigned I, const unsigned P1, const unsigned P2>
void foo_ptr(unsigned ii) {
  printf("Running host version...\n");

  unsigned *A = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  unsigned *B = (unsigned*)malloc(I*I*I*sizeof(unsigned));
  
  for (unsigned i = 0; i<I*I*I; ++i) {
    A[i] = B[i] = ii + i + 0;
  }
  
  for (unsigned i = 0; i<I; ++i)
  {
    unsigned R1[P1];
    for (unsigned r = 0; r<P1; ++r)
      R1[r] = ii + r + 5;
    
    Arr(A,i,0,0) += ii;
    Arr(B,i,0,0) += ii;
    
    for (unsigned j=0; j<I; ++j)
    {
      unsigned r1 = 0;
      for (unsigned r=0; r<P1; ++r)
        r1 += R1[r];
        
      unsigned R2[P2];
      for (unsigned r = 0; r<P2; ++r)
        R2[r] = ii + r + 7;
        
      Arr(A,i,j,0) += r1;
      Arr(B,i,j,0) += r1;
        
      for (unsigned k = 0; k<I; ++k)
      {
        unsigned r2 = 0;
        for (unsigned r=0; r<P2; ++r)
          r2 += R2[r];
          
        Arr(A,i,j,k) += r2;
        Arr(B,i,j,k) += r2;
      }
    }
  }
  
  printf("Done with host version...\n");
  
  for(unsigned i = 0; i<I*I*I; ++i) {
    if (A[i] != B[i]) {
      printf("Error (host) A[i] != B[i] at %08x: %08x != %08x\n", i, A[i], B[i]);
      break;
    }
  }
  
  foo_ptr_device_parallel<I,P1,P2>         (ii, &A[0], &B[0]);
  foo_ptr_device_parallel_parallel<I,P1,P2>(ii, &A[0], &B[0]);
  foo_ptr_device_simd<I,P1,P2>             (ii, &A[0], &B[0]);
  foo_ptr_device_parallel_simd<I,P1,P2>    (ii, &A[0], &B[0]);
  free(A);
  free(B);
}

template<typename T>
void loop_sched(void) {
  T foo = 2;
  T cpuExec[2] = {0,0};
  #pragma omp target
  #pragma omp parallel for if(foo) num_threads(foo) schedule(static,foo)
  for (int i = 0; i < 100; i++)
  {
    cpuExec[omp_get_thread_num()] += i;
  }

  printf("--> 2425=%ld 2525=%ld\n",(long)cpuExec[0], (long)cpuExec[1]);
}

void static_data_sharing(void) {
  static int temp[2] = { 1, 1 };
  int A[2] = { 1, 1 };

  #pragma omp target map(tofrom: A[0:2], temp)
  {
    #pragma omp parallel for num_threads(2)
    for (int i=0;i<50;i++) {
      A[omp_get_thread_num()] += 1;
      temp[omp_get_thread_num()] += 2;
    }
    #pragma omp barrier
  }

  printf("--> 26=%d 26=%d\n",A[0],A[1]);
  printf("--> 51=%d 51=%d\n",temp[0],temp[1]);
}

int main(void) {
  unsigned ii = 1;

  printf("Testing arrays...\n");

  check_offloading();
  foo<16,16,16>(ii);
  foo<32,16,16>(ii);
  foo<32,32,16>(ii);
  foo<32,32,32>(ii);
  foo<32,32,32>(ii);
  foo<64,32,32>(ii);
  foo<64,64,32>(ii);
  foo<64,64,44>(ii);
  foo<67,32,32>(ii);
  foo<67,67,32>(ii);
  foo<67,67,47>(ii);
  foo<80,32,32>(ii);
  foo<80,33,32>(ii); 
  foo<80,33,33>(ii); 

  printf("Testing ptrs...\n");

  foo_ptr<16,16,16>(ii);
  foo_ptr<32,16,16>(ii);
  foo_ptr<32,32,16>(ii);
  foo_ptr<32,32,32>(ii);
  foo_ptr<32,32,32>(ii);
  foo_ptr<64,32,32>(ii);
  foo_ptr<64,64,32>(ii);
  foo_ptr<64,64,44>(ii);
  foo_ptr<67,32,32>(ii);
  foo_ptr<67,67,32>(ii);
  foo_ptr<67,67,47>(ii);
  foo_ptr<80,32,32>(ii);
  foo_ptr<80,33,32>(ii);
  foo_ptr<80,33,33>(ii);
  foo_ptr<129,33,33>(ii);

  printf("Testing loop schedule capture...\n");
  loop_sched<int>();
  loop_sched<long>();

  printf("Testing static var data sharing...\n");
  static_data_sharing();

  printf("End!\n");
  return 0;
}
