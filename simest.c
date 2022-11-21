#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

#define TIME_START start = clock();
#define TIME_END(NAME)     \
    end = clock(); \
    printf(NAME);          \
    printf(" takes %fs\n", (float)(end - start)/CLOCKS_PER_SEC);

#define SIZE 64000

int main() {
    time_t start;
    time_t end;
    size_t M = SIZE;
    size_t N = SIZE;
    size_t K = SIZE;
    float alpha = 1;
    float beta = 0;
    size_t lda = M;
    size_t ldb = K;
    size_t ldc = N;
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(N * K * sizeof(float));
    for (size_t i = 0; i < K * M; i++) {
        A[i] = i;
    }
    for (size_t i = 0; i < K * N; i++) {
        B[i] = i;
    }
    float* C = (float*)malloc(N * M * sizeof(float));
    TIME_START
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    TIME_END("BLAS")
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", C[i * N + j]);
    //     }
    //     printf("\n");
    // }
}

// int main(){

//        int n;                          /*! array size */
//        double da;                      /*! double constant */
//        double *dx;                     /*! input double array */
//        int incx;                        /*! input stride */
//        double *dy;                      /*! output double array */
//        int incy;                       /*! output stride */

//        int i;

//        n = 10;
//        da = 10;
//        dx = (double*)malloc(sizeof(double)*n);
//        incx = 1;
//        dy = (double*)malloc(sizeof(double)*n);
//        incy = 1;

//        for(i=0;i<n;i++){
//                dx[i] = 9-i;
//                dy[i] = i;
//                printf("%f ",dy[i]);
//        }
//        printf("\n");

//        cblas_daxpy(n, da, dx,incx, dy, incy);
//    //    cblas_dcopy(n, dx,incx, dy, incy);

//        for(i=0;i<n;i++){
//            printf("%f ",dy[i]);
//        }
//        printf("\n");

//        return 0;
// }