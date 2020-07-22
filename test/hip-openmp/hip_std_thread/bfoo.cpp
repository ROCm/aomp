#include <stdio.h>
#include <iostream>
#include <thread>


void* wrapper(void * start) {
    fprintf(stderr, "info: synchronize\n");
    return NULL;
}

long inc=0;
void doit() {
	wrapper((long*)++inc);
}

int main(int argc, char* argv[]) {
  int nThreads = 10;
  if (argc > 1) nThreads = atoi(argv[1]);
  fprintf(stderr, "using %d threads\n",nThreads);

  std::thread tID[nThreads];

  for(long i=0;i<nThreads;i++)
    tID[i]=std::thread(doit);
  for(long i=0;i<nThreads;i++)
	tID[i].join();
  return 0;
}

