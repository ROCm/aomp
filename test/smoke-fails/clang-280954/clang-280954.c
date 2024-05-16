#include <stdio.h>
#include <assert.h>

int main() {
    int a[1000];

#pragma omp target teams distribute
    for (int i = 0 ; i < 1000 ; i++) {
        a[i] = i;
    }
    
    for (int i = 0 ; i < 1000 ; i++) {
        assert( a[i] == i );
    }
    printf("PASS\n");
    return 0;
}
