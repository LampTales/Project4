#include <stddef.h>
#pragma once

// A matrix lib that based on float numbers.
struct Matrix {
    size_t row;
    size_t col;
    float* arr;
};

int oldMul(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

int matmul_plain(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

int matmul_improved(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

int matmul_improvedMP(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

int matmul_improvedDIV(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

void innerMul(float* p1, float* p2, float* ans, size_t SIZE);

struct Matrix* createMatrix(size_t row, size_t col);

struct Matrix* createMatrixWithIni(size_t row, size_t col, const float* fpointer);

struct Matrix* createTestMatrix(size_t row, size_t col);

void deleteMatrix(struct Matrix** mpp);

int printMatrix(const struct Matrix* mpointer);
