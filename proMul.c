#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "cblas.h"

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
    size_t offset = (8 - (mp1->col % 8)) % 8;
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
    size_t offset = (8 - (mp1->col % 8)) % 8;
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
#ifdef __AVX2__
    printf("From SIMD: AVX ON\n");
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
    size_t offset = (8 - (mp1->col % 8)) % 8;
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
    // printf("core num: %d\n", omp_get_num_procs());

    // using SIMD
#ifdef __AVX2__
    printf("From OMP: AVX2 ON\n");
    // omp_set_num_threads(20);
    __m256 a, b, c;
    float sum[8] = {0};
#pragma omp parallel for private(a, b, c, sum)
    for (size_t i = 0; i < answer->row; i++) {
        // printf("here is %d", omp_get_thread_num());
        size_t asp = i * offsetLen;
        for (size_t j = 0; j < answer->col; j++) {
            __m256 c = _mm256_setzero_ps();
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
    return 0;
#endif

    free(p1);
    free(p2);
    return 1;
}

int matmul_improvedDIV(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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

    size_t blockSize = 512;
    size_t blockArea = blockSize * blockSize;

    size_t AblockRow = mp1->row / blockSize;
    size_t AblockCol = mp1->col / blockSize;
    float* p1 = (float*)aligned_alloc(256, (mp1->row) * (mp1->col) * sizeof(float));
    size_t ai = 0;
    for (size_t rbc = 0; rbc < AblockRow; rbc++) {
        for (size_t cbc = 0; cbc < AblockCol; cbc++) {
            size_t spot1 = rbc * AblockCol * blockArea + cbc * blockSize;
            for (size_t smr = 0; smr < blockSize; smr++) {
                size_t spot2 = smr * mp1->col;
                for (size_t x = 0; x < blockSize; x++) {
                    p1[ai] = mp1->arr[spot1 + spot2 + x];
                    ai++;
                }
            }
        }
    }

    size_t BblockRow = mp2->row / blockSize;
    size_t BblockCol = mp2->col / blockSize;
    float* p2 = (float*)aligned_alloc(256, (mp2->row) * (mp2->col) * sizeof(float));
    size_t bi = 0;
    for (size_t cbc = 0; cbc < BblockCol; cbc++) {
        for (size_t rbc = 0; rbc < BblockRow; rbc++) {
            size_t spot1 = rbc * BblockCol * blockArea + cbc * blockSize;
            for (size_t smc = 0; smc < blockSize; smc++) {
                for (size_t x = 0; x < blockSize; x++) {
                    p2[bi] = mp2->arr[spot1 + smc + mp2->col * x];
                    bi++;
                }
            }
        }
    }

    for (size_t i = 0; i < answer->row * answer->col; i++) {
        answer->arr[i] = 0;
    }

#ifdef __AVX2__
    printf("From DIV: AVX2 ON\n");
    float* store = (float*)malloc(blockArea * sizeof(float));
    for (size_t i = 0; i < AblockRow; i++) {  // every block row
        // printf("i = %ld\n", i);
        for (size_t j = 0; j < BblockCol; j++) {  // every block col
            for (size_t bcnt = 0; bcnt < AblockCol; bcnt++) {
                innerMul(p1 + (i * AblockCol + bcnt) * blockArea, p2 + (j * BblockRow + bcnt) * blockArea, store, blockSize);
                size_t spot = 0;
                size_t spotcnt1 = i * BblockCol * blockArea + j * blockSize;
                for (size_t x = 0; x < blockSize; x++) {
                    size_t spotcnt2 = x * answer->col;
                    for (size_t y = 0; y < blockSize; y++) {
                        answer->arr[spotcnt1 + spotcnt2 + y] += store[spot];
                        spot++;
                    }
                }
            }
        }
    }
    free(store);
#else
    printf("AVX2 is not supported\n");
    free(p1);
    free(p2);
    return 0;
#endif
    free(p1);
    free(p2);
    return 1;
}

inline void innerMul(float* p1, float* p2, float* ans, size_t SIZE) {
#ifdef WITH_AVX2
    __m256 a, b, c;
    float sum[8] = {0};
#pragma omp parallel for private(a, b, c, sum)
    for (size_t i = 0; i < SIZE; i++) {
        // printf("here is %d", omp_get_thread_num());
        size_t asp = i * SIZE;
        for (size_t j = 0; j < SIZE; j++) {
            __m256 c = _mm256_setzero_ps();
            size_t bsp = j * SIZE;
            for (size_t cnt = 0; cnt < SIZE; cnt += 8) {
                a = _mm256_loadu_ps(p1 + asp + cnt);
                b = _mm256_loadu_ps(p2 + bsp + cnt);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(sum, c);
            ans[i * SIZE + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }
#else
    printf("AVX2 is not supported\n");
#endif
}

int matmul_BLAS(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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

    size_t M = mp1->row;
    size_t K = mp1->col;
    size_t N = mp2->col;
    float alpha = 1;
    float beta = 0;
    size_t lda = M;
    size_t ldb = K;
    size_t ldc = N;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, mp1->arr, lda, mp2->arr, ldb, beta, answer->arr, ldc);
}