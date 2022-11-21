#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Matrix.h"

#define TIME_START start = omp_get_wtime();
#define TIME_END(NAME)     \
    end = omp_get_wtime(); \
    printf(NAME);          \
    printf(" takes %fs\n", (float)(end - start));

#define SIZE 16

// printf(" takes %fs\n", (float)(end - start)/CLOCKS_PER_SEC);

int main() {
    double start;
    double end;

    struct Matrix* x = createMatrix(1, 1);
    struct Matrix* y = createMatrix(1, 1);
    struct Matrix* z = createMatrix(1, 1);
    struct Matrix* t = createMatrix(1, 1);

    TIME_START
    struct Matrix* m1 = createTestMatrix(SIZE, SIZE);
    struct Matrix* m2 = createTestMatrix(SIZE, SIZE);
    TIME_END("create matrix")
    
    // TIME_START
    // oldMul(m1, m2, y);
    // TIME_END("oldMul")
    // printMatrix(y);
    // printf("\n");
    // TIME_START
    // matmul_plain(m1, m2, x);  // this is legal
    // TIME_END("Plain")
    // printMatrix(x);
    // printf("\n");
    // TIME_START
    // matmul_improved(m1, m2, z);
    // TIME_END("SIMD")
    // printMatrix(z);

    TIME_START
    matmul_improvedMP(m1, m2, t);
    TIME_END("openMP")
    printMatrix(t);
    TIME_START
    matmul_improvedMP(m1, m2, t);
    TIME_END("openMP")

    TIME_START
    matmul_improvedMP(m1, m2, t);
    TIME_END("openMP")

    deleteMatrix(&m1);
    deleteMatrix(&m2);
    deleteMatrix(&x);
    deleteMatrix(&y);
    deleteMatrix(&z);
    deleteMatrix(&t);
}