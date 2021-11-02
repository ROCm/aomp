// mimic NWChem tgt_sd_t_s1_1 kernel but in C

#include <stdio.h>

void sd_t_s1_1(int l1, int l2, int l3, int l4, int l5, int l6,
	       int u1, int u2, int u3, int u4, int u5, int u6) {
  float a[24][24][24][24][24][24];
  float b[24][24][24][24][24][24];

  int i1, i2, i3, i4, i5, i6;
  
  for (i1 = 0; i1 < 24; ++i1)
    for (i2 = 0; i2 < 24; ++i2)
      for (i3 = 0; i3 < 24; ++i3)
	for (i4 = 0; i4 < 24; ++i4)
	  for (i5 = 0; i5 < 24; ++i5)
	    for (i6 = 0; i6 < 24; ++i6) a[i1][i2][i3][i4][i5][i6] = 3.0;

#pragma omp target teams distribute parallel for collapse(6)
  for (i1 = l1; i1 < u1; ++i1)
    for (i2 = l2; i2 < u2; ++i2)
      for (i3 = l3; i3 < u3; ++i3)
	for (i4 = l4; i4 < u4; ++i4)
	  for (i5 = l5; i5 < u5; ++i5)
	    for (i6 = l6; i6 < u6; ++i6)
	      b[i1][i2][i3][i4][i5][i6] = a[i1][i2][i3][i4][i5][i6] + i3;

  for (i1 = l1; i1 < 4; ++i1) {
    for (i2 = l2; i2 < 4; ++i2)
      printf("%f ", b[i1][i2][1][1][1][1]);
    printf("\n");
  }
}
  
int main() {
  sd_t_s1_1(0, 0, 0, 0, 0, 0, 24, 24, 24, 24, 24, 24);
}
