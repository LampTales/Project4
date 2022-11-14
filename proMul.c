#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"
#include "innerTools.h"

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

float dotproduct_avx2(const float *p1, const float * p2, size_t n)
{
#ifdef WITH_AVX2
    if(n % 8 != 0)
    {
        printf("The size n must be a multiple of 8.\n");
        return 0.0f;
    }

    float sum[8] = {0};
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i+=8)
    {
        a = _mm256_load_ps(p1 + i);
        b = _mm256_load_ps(p2 + i);
        c =  _mm256_add_ps(c, _mm256_mul_ps(a, b));
    }
    _mm256_store_ps(sum, c);
    return (sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]);
#else
    printf("AVX failed.\n");
        return 0.0f;
#endif
}

int matmul_plain(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
    return multiplyMatrix(mp1, mp2, answer);
}

int matmul_improved(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
    // reporting or fixing the illegal cases
    int errCode = 110;
    if (isLegalMatrixWithData(mp1) == 0 || isLegalMatrixWithData(mp2) == 0) {
        return errCode + 5;
    } else if (mp1->col != mp2->row) {
        return errCode + 6;
    } else if (isLegalMatrix(answer) == 0) {
        return errCode + 7;
    } else if (mp1 == answer || mp2 == answer) {
        return errCode + 8;
    }
    if (answer->row != mp1->row || answer->col != mp2->col) {
        answer->row = mp1->row;
        answer->col = mp2->col;
        if (answer->arr != NULL) {
            free(answer->arr);
            answer->arr = NULL;
        }
    }
    if (answer->arr == NULL) {
        float* fpo = (float*)malloc(answer->row * answer->col * sizeof(float));
        answer->arr = fpo;
    }

    // using SIMD
    #ifdef WITH_AVX2
    if(n % 8 != 0)
    {
        printf("The size n must be a multiple of 8.\n");
        return 0.0f;
    }

    float sum[8] = {0};
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i+=8)
    {
        a = _mm256_load_ps(p1 + i);
        b = _mm256_load_ps(p2 + i);
        c =  _mm256_add_ps(c, _mm256_mul_ps(a, b));
    }
    _mm256_store_ps(sum, c);
    return (sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]);
#else
    printf("AVX2 is not supported\n");
    return 0.0;
#endif
}