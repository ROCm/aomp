#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
#pragma omp target
  ;

  void *Handle = dlopen(argv[1], RTLD_NOW);
  int (*Foo)(void);

  if (Handle == NULL) {
    printf("dlopen() failed: %s\n", dlerror());
    return 1;
  }
  Foo = (int (*)(void))dlsym(Handle, "foo");
  if (Handle == NULL) {
    printf("dlsym() failed: %s\n", dlerror());
    return 1;
  }

  return Foo();
}

// CHECK: DONE.
// CHECK-NOT: {{abort|fault}}
