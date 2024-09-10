#include <stdio.h>
#include <omp.h>

int main(int argc, const char *argv[]) {
  int array_test[10][10];
  int thread_num[10][10];
  int team_num[10][10];
  int chk = 0;

  {
    for (int i =0; i < 10; ++i) {
      for (int j =0; j < 10; ++j) {
        array_test[i][j] = 999;
        team_num[i][j] = -1;
        thread_num[i][j] = -1;
      }
    }
  }

  #pragma omp target teams distribute
  for (int i =0; i < 10; ++i) {
    #pragma omp parallel for
    for (int j =0; j < 10; ++j) {
      array_test[i][j] = i + j + 2;
      team_num[i][j] = omp_get_team_num();
      thread_num[i][j] = omp_get_thread_num();
    }
  }

  printf("%6s", "");
  for (int j =0; j < 10; ++j) {
    printf("%13d", j+1);
  }
  printf("\n");
  printf("%6s", "");
  for (int j =0; j < 10; ++j) {
    printf("%13s", "-----------");
  }
  printf("\n");

  for (int i =0; i < 10; ++i) {
    printf("%5d|", i+1);
    for (int j =0; j < 10; ++j) {
      chk += array_test[i][j];
      printf("%5d", array_test[i][j]);
      printf(":%4d:%2d", team_num[i][j], thread_num[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  int exp = 2*10*10 + (10-1)*10*10;
  printf("exp %5d\n", exp);
  printf("chk %5d\n", chk);
  if (chk != exp) {
      printf("FAIL\n");
      return 1;
  }

  printf("PASS\n");
  return 0;
}
