#include "two.h"

#pragma omp declare target

//static
int state = 2;

int two(void)
{
	return state;
}

#pragma omp end declare target
