#include "Matrix.h"
#include <stdio.h>
#include <stdlib.h>

struct Matrix* createMatrix(size_t row, size_t col) {
    if (row == 0 || col == 0) {
        return NULL;
    }

    float* fpo = (float*)malloc(row * col * sizeof(float));
    for (size_t i = 0; i < row * col; i++) {
        fpo[i] = 0;
    }
    struct Matrix* mpo = (struct Matrix*)malloc(sizeof(struct Matrix));
    mpo->row = row;
    mpo->col = col;
    mpo->arr = fpo;
    return mpo;
}

struct Matrix* createTestMatrix(size_t row, size_t col) {
    if (row == 0 || col == 0) {
        return NULL;
    }

    float* fpo = (float*)malloc(row * col * sizeof(float));
    for (size_t i = 0; i < row * col; i++) {
        fpo[i] = i;
    }
    struct Matrix* mpo = (struct Matrix*)malloc(sizeof(struct Matrix));
    mpo->row = row;
    mpo->col = col;
    mpo->arr = fpo;
    return mpo;
}

struct Matrix* createMatrixWithIni(size_t row, size_t col, const float* fpointer) {
    if (row == 0 || col == 0 || fpointer == NULL) {
        return NULL;
    }

    float* fpo = (float*)malloc(row * col * sizeof(float));
    for (size_t i = 0; i < row * col; i++) {
        fpo[i] = fpointer[i];
    }
    struct Matrix* mpo = (struct Matrix*)malloc(sizeof(struct Matrix));
    mpo->row = row;
    mpo->col = col;
    mpo->arr = fpo;
    return mpo;
}

void deleteMatrix(struct Matrix** mpp) {
    if (mpp == NULL) {
        return;
    }
    if ((*mpp) == NULL) {
        return;
    }
    if ((*mpp)->arr != NULL) {
        free((*mpp)->arr);
    }
    free((*mpp));
    *mpp = NULL;
    return;
}

int printMatrix(const struct Matrix* mpointer) {
    if (mpointer == NULL || mpointer->row == 0 || mpointer->col == 0 || mpointer->arr == NULL) {
        return 51;
    }
    for (size_t i = 0; i < (mpointer->row) * (mpointer->col); i++) {
        printf("%f\t", mpointer->arr[i]);
        if ((i + 1) % mpointer->col == 0) {
            printf("\n");
        }
    }
    return 50;
}

void printError(int errCode) {
    
}