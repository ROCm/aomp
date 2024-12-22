//#include <stdio.h>

int main(int argc, char **argv) {
#ifdef __CUDA_ARCH__
#error CUDA_ARCH is set!!
#endif
  return 0;
}
