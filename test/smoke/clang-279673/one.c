#include "two.h"
#include <stdio.h>

int main(void)
{
	#pragma omp target
	printf("%d\n", two() - 1);
	return 0;
}
