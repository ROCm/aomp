#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp target
  {
    int num;
    printf ("Signed: %i\n", 392);
    printf ("Unsigned: %d\n", 7235);
    printf ("Unsigned: %o\n", 0610);
    printf ("Unsigned: %x\n", 0x7fa);
    printf ("Unsigned: %X\n", 0x7fa);
    printf ("floating point: %f\n", 392.65);
    printf ("floating point: %F\n", 392.65);
    printf ("S notation point: %e\n", 3.9265e+2);
    printf ("S notation point: %E\n", 3.9265e+2);
    printf ("floating point: %g\n", 392.65);
    printf ("floating point: %G\n", 392.65);
    printf ("Hexadecimal: %a\n", 3.9265e+2);
    printf ("Hexadecimal: %A\n", 3.9265e+2);
    printf ("Characters: %c\n", 65);
    printf ("Pointer: %p\n", &num);
    printf ("Signed: %d\n", 392);
    printf ("Only percent sign %%\n");

    printf ("No specifier!\n");
  }

  return 0;
}

