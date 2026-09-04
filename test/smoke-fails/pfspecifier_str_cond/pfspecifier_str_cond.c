#include <stdio.h>
#include <string.h>
#include <omp.h>

#pragma omp declare target
int f(int a)
{
  return a/10;
}
#pragma omp end declare target

int main()
{
    int i;
    i=f(2200);
    char *str_begin = "<<<";
    char *str_end = ">>>";
    char *str_alt = "67890_XYZxyz?!";
    char *str = "12345_ABCDEFGabcdegf?!";
    printf("String on   host: %s\n", "12345_ABCDEFGabcdegf?!");

#pragma omp target map(to: str_begin[0:strlen(str_begin)+1],\
  str_end[0:strlen(str_end)+1], str[0:strlen(str)+1],\
  str_alt[0:strlen(str_alt)+1])
  {
    char *fmt = "String on device: %s\n";
    printf("String on device: %s\n", "12345_ABCDEFGabcdegf?!");
    printf("String on device: %s\n", str);
    printf(fmt, str);
// Choose string to print
    char *outlined_conditional_1 = f(1) ? str_alt : str;
    printf(fmt, outlined_conditional_1);
    char *outlined_conditional_2 = f(10) ? str_alt : str;
    printf(fmt, outlined_conditional_2);
// Choose string format size using variable, string will be right aligned
    char *outlined_conditional_3 = f(10) ? str_alt : str;
    printf("String on device: %*s\n", f(220), outlined_conditional_3);
// Choose maximium string format size using variable
// Currently doesn't work on Nvidia
//    printf ("String on device: %.*s\n", f(70), f(1) ? str_alt : str);
    char *outlined_conditional_4 = f(10) ? str_alt : str;
    printf("Data   on device: %2d %s%*s%s %2d%*d\n", f(i), str_begin, f(i),
      outlined_conditional_4, str_end, f(i), f(50), f(i));
  }
  char *outlined_conditional_5 = f(10) ? str_alt : str;
  printf("Data   on   host: %2d %s%*s%s %2d%*d\n", f(i), str_begin, f(i),
    outlined_conditional_5, str_end, f(i), f(50), f(i));

  return 0;
}
