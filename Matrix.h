#include <stddef.h>
#pragma once

// A matrix lib that based on float numbers.
struct Matrix {
    size_t row;
    size_t col;
    float* arr;
};

float dotproduct_avx2(const float* p1, const float* p2, size_t n);

int matmul_plain(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

int matmul_improved(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

struct Matrix* createMatrix(size_t row, size_t col);

struct Matrix* createMatrixWithIni(size_t row, size_t col, const float* fpointer);

void deleteMatrix(struct Matrix** mpp);

int printMatrix(const struct Matrix* mpointer);
