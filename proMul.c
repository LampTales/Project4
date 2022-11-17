#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

float dotproduct_avx2(const float* p1, const float* p2, size_t n) {
#ifdef WITH_AVX2
    if (n % 8 != 0) {
        printf("The size n must be a multiple of 8.\n");
        return 0.0f;
    }

    float sum[8] = {0};
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        a = _mm256_loadu_ps(p1 + i);
        b = _mm256_loadu_ps(p2 + i);
        c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
    }
    _mm256_storeu_ps(sum, c);
    return (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]);
#else
    printf("AVX failed.\n");
    return 0.0f;
#endif
}

int matmul_plain(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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

    size_t length = (mp1->row) * (mp2->col);
    for (size_t i = 0; i < length; i++) {
        answer->arr[i] = 0;
        size_t aspot = i / mp2->col;
        size_t bspot = i - aspot * mp2->col;
        for (size_t x = 0; x < mp1->col; x++) {
            answer->arr[i] += mp1->arr[aspot * mp1->col + x] * mp2->arr[x * mp2->col + bspot];
        }
    }
    return 110;
}

int matmul_improved(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
    // reporting or fixing the illegal cases
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

    // loading data
    size_t offset = 8 - (mp1->col % 8);
    size_t offsetLen = offset + mp1->col;
    float* p1 = (float*)aligned_alloc(256, (mp1->row) * offsetLen * sizeof(float));
    for (size_t i = 0; i < mp1->row; i++) {
        size_t startO = i * (mp1->col);
        size_t startN = i * offsetLen;
        for (size_t j = 0; i < mp1->col; j++) {
            p1[startN + j] = mp1->arr[startO + j];
        }
        for (size_t j = mp1->col; j < offsetLen; j++) {
            p1[startN + j] = 0;
        }
    }
    float* p2 = (float*)aligned_alloc(256, (mp1->row) * offsetLen * sizeof(float));
    for (size_t i = 0; i < mp1->row; i++) {
        size_t cntO = 0;
        size_t startN = i * offsetLen;
        for (size_t j = 0; j < mp1->col; j++)
        {
            /* code */
        }
        
    }

    // using SIMD
#ifdef WITH_AVX2
    if (n % 8 != 0) {
        printf("The size n must be a multiple of 8.\n");
        return 0.0f;
    }

    float sum[8] = {0};
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        a = _mm256_load_ps(p1 + i);
        b = _mm256_load_ps(p2 + i);
        c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
    }
    _mm256_store_ps(sum, c);
    return (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]);
#else
    printf("AVX2 is not supported\n");
    return 0.0;
#endif

    return 110;
}