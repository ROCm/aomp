#include <omp.h>
#include <cstdio>

#define NUM 10

#pragma omp begin declare target
#ifdef __AMDGCN__
int myfn(int IsSPMD, int threadId) {
  return __builtin_amdgcn_workitem_id_x() == threadId;
}
void *fns[2] = { (void*)myfn, nullptr };

int test_fn(void *fn, int IsSPMD, int threadId) {
  return ((int (*)(int IsSPMD, int threadId))(fn))(IsSPMD, threadId);
}

int run(int tid, int f) {
  return test_fn(fns[f], 1, tid);
}

#else
int __builtin_amdgcn_workitem_id_x() { return -1; }
int run(int, int) { return 0; }
#endif

#pragma omp end declare target

int main() {
  int *b = new int[NUM];
  int f = 0;
#pragma omp target parallel for map(tofrom:b[0:NUM])
  for (int i = 0; i < NUM; i++) {
    int tid = __builtin_amdgcn_workitem_id_x();
    b[i] = run(tid, 0);
  }
  bool allOne = true;
  for (int i = 0; i < NUM; i++) {
    allOne = allOne && b[i];
    printf("%d ", b[i]);
  }

  printf("\n%s\n", allOne ? "SUCCESS" : "FAILED");
  return !allOne;
}
