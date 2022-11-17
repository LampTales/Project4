#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "Matrix.h"

int main() {
    // struct Matrix* x = createMatrix(1, 1);
    // struct Matrix* y = createMatrix(1, 0); // this is illegal
    // if (y == NULL) { // this should be true
    //     printf("y is not created\n");
    // }

    // float array1[] = {1, 2, 3, 4, 5, 6};
    // float array2[] = {2, 3, 4, 5, 6, 7};
    // float array3[] = {1, 2, 3, 4};
    // float* fp1 = array1;
    // float* fp2 = array2;
    // float* fp3 = array3;
    // struct Matrix* m1 = createMatrixWithIni(2, 3, fp1);
    // struct Matrix* m2 = createMatrixWithIni(3, 2, fp2);
    // struct Matrix* m3 = createMatrixWithIni(2, 2, fp3);

    // float ans = 0;
    // float* ap = &ans;
    // getMinimum(m1, ap);
    // printf("maximum = %f\n", ans);
    // printError(setElement(m1, 7, 8, 0.5));  // this is illegal
    // printError(setElement(m1, 1, 2, 0.5));  // this is legal
    // getMinimum(m1, ap);
    // printf("maximum = %f\n", ans);

    // struct Matrix* m4 = copyMatrix(m1);
    // printError(setElement(m4, 1, 2, 1234));
    // printError(getElement(m4, 1, 2, ap));
    // printf("element in m4 = %f\n", ans);
    // printError(getElement(m1, 1, 2, ap));
    // printf("element in m1 = %f\n", ans);  // the two element should not be the same, as it is deep copy

    // printf("m2 before addScalar:\n");
    // printMatrix(m2);
    // printError(addScalar(m2, 2));
    // printf("m2 after addScalar:\n");
    // printMatrix(m2);

    // printf("Test for matrix operation:\n");
    // printError(multiplyMatrix(m1, m3, x));   // this is illegal
    // printError(multiplyMatrix(m1, m2, m2));  // this is illegal
    // printError(multiplyMatrix(m1, m2, x));   // this is legal
    // printMatrix(x);

    // deleteMatrix(&m1);
    // deleteMatrix(&m2);
    // deleteMatrix(&m3);
    // deleteMatrix(&m4);
    // deleteMatrix(&x);
    // if (x == NULL) { // this should be true
    //     printf("x is NULL after delete\n");
    // }
    // return 0;

    size_t nSize = 8;
    //float * p1 = new float[nSize](); //the memory is not aligned
    //float * p2 = new float[nSize](); //the memory is not aligned

    //256bits aligned, C++17 standard
    float * p1 = (float*)aligned_alloc(256, nSize*sizeof(float));
    float * p2 = (float*)aligned_alloc(256, nSize*sizeof(float));
    p1[2] = 2.3f;
    p2[2] = 3.0f;
    p1[nSize-1] = 2.0f;
    p2[nSize-1] = 1.1f;
    // float * p2 = (float*)malloc(256 * nSize * sizeof(float));
    
    float result = 0.0f;

    // TIME_START
    result = dotproduct_avx2(p1, p2, nSize);
    // TIME_END("SIMD")
    printf("%f\n", result);

    free(p1);
    free(p2);
}