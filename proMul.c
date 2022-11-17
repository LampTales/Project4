#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

#include <omp.h>

int oldMul(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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

// not using SIMD
int matmul_plain(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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
    // printf("loading data for p1\n");
    size_t offset = 8 - (mp1->col % 8);
    size_t offsetLen = offset + mp1->col;
    float* p1 = (float*)aligned_alloc(256, (mp1->row) * offsetLen * sizeof(float));
    for (size_t i = 0; i < mp1->row; i++) {
        size_t startO = i * (mp1->col);
        size_t startN = i * offsetLen;
        for (size_t j = 0; j < mp1->col; j++) {
            p1[startN + j] = mp1->arr[startO + j];
        }
        for (size_t j = mp1->col; j < offsetLen; j++) {
            p1[startN + j] = 0;
        }
    }
    // printf("loading data for p2\n");
    float* p2 = (float*)aligned_alloc(256, (mp2->col) * offsetLen * sizeof(float));
    for (size_t i = 0; i < mp2->col; i++) {
        size_t cntO = i;
        size_t startN = i * offsetLen;
        for (size_t j = 0; j < mp2->row; j++) {
            p2[startN + j] = mp2->arr[cntO];
            cntO += mp2->col;
        }
        for (size_t j = mp2->row; j < offsetLen; j++) {
            p2[startN + j] = 0;
        }
    }

    // calculating
    // printf("calculating\n");
    for (size_t i = 0; i < answer->row; i++) {
        for (size_t j = 0; j < answer->col; j++) {
            size_t spot = i * answer->row + j;
            size_t asp = i * offsetLen;
            size_t bsp = j * offsetLen;
            answer->arr[spot] = 0;
            for (size_t cnt = 0; cnt < offsetLen; cnt++) {
                answer->arr[spot] += p1[asp + cnt] * p2[bsp + cnt];
            }
        }
    }

    free(p1);
    free(p2);
    return 0;
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
        for (size_t j = 0; j < mp1->col; j++) {
            p1[startN + j] = mp1->arr[startO + j];
        }
        for (size_t j = mp1->col; j < offsetLen; j++) {
            p1[startN + j] = 0;
        }
    }
    float* p2 = (float*)aligned_alloc(256, (mp2->col) * offsetLen * sizeof(float));
    for (size_t i = 0; i < mp2->col; i++) {
        size_t cntO = i;
        size_t startN = i * offsetLen;
        for (size_t j = 0; j < mp2->row; j++) {
            p2[startN + j] = mp2->arr[cntO];
            cntO += mp2->col;
        }
        for (size_t j = mp2->row; j < offsetLen; j++) {
            p2[startN + j] = 0;
        }
    }

    // using SIMD
#ifdef WITH_AVX2
    printf("AVX2 ON\n");
    for (size_t i = 0; i < answer->row; i++) {
        for (size_t j = 0; j < answer->col; j++) {
            float sum[8] = {0};
            __m256 a, b;
            __m256 c = _mm256_setzero_ps();
            size_t asp = i * offsetLen;
            size_t bsp = j * offsetLen;
            for (size_t cnt = 0; cnt < offsetLen; cnt += 8) {
                a = _mm256_loadu_ps(p1 + asp + cnt);
                b = _mm256_loadu_ps(p2 + bsp + cnt);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(sum, c);
            answer->arr[i * answer->row + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }
#else
    printf("AVX2 is not supported\n");
    free(p1);
    free(p2);
    return 0.0;
#endif

    free(p1);
    free(p2);
    return 110;
}

int matmul_improvedMP(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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
        for (size_t j = 0; j < mp1->col; j++) {
            p1[startN + j] = mp1->arr[startO + j];
        }
        for (size_t j = mp1->col; j < offsetLen; j++) {
            p1[startN + j] = 0;
        }
    }
    float* p2 = (float*)aligned_alloc(256, (mp2->col) * offsetLen * sizeof(float));
    for (size_t i = 0; i < mp2->col; i++) {
        size_t cntO = i;
        size_t startN = i * offsetLen;
        for (size_t j = 0; j < mp2->row; j++) {
            p2[startN + j] = mp2->arr[cntO];
            cntO += mp2->col;
        }
        for (size_t j = mp2->row; j < offsetLen; j++) {
            p2[startN + j] = 0;
        }
    }
    printf("core num: %d\n", omp_get_num_procs());

    // using SIMD
#ifdef WITH_AVX2
    printf("AVX2 ON\n");
    omp_set_num_threads(8);
#pragma omp parallel for
    for (size_t i = 0; i < answer->row; i++) {
        for (size_t j = 0; j < answer->col; j++) {
            float sum[8] = {0};
            __m256 a, b;
            __m256 c = _mm256_setzero_ps();
            size_t asp = i * offsetLen;
            size_t bsp = j * offsetLen;
            for (size_t cnt = 0; cnt < offsetLen; cnt += 8) {
                a = _mm256_loadu_ps(p1 + asp + cnt);
                b = _mm256_loadu_ps(p2 + bsp + cnt);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(sum, c);
            answer->arr[i * answer->row + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }

#else
    printf("AVX2 is not supported\n");
    free(p1);
    free(p2);
    return 0.0;
#endif

    free(p1);
    free(p2);
    return 110;
}