#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"

struct Matrix* createRawMatrix(int row, int col) {
    if (row <= 0 || col <= 0) {
        return NULL;
    }
    float* fpo = (float*)malloc(row * col * sizeof(float));
    struct Matrix* mpo = (struct Matrix*)malloc(sizeof(struct Matrix));
    mpo->row = row;
    mpo->col = col;
    mpo->arr = fpo;
    return mpo;
}

int isLegalMatrix(const struct Matrix* mpointer) {
    if (mpointer == NULL || mpointer->row <= 0 || mpointer->col <= 0) {
        return 0;
    }
    return 1;
};

int isLegalMatrixWithData(const struct Matrix* mpointer) {
    if (isLegalMatrix(mpointer) == 0 || mpointer->arr == NULL) {
        return 0;
    }
    return 1;
};

int hasSuchSpot(const struct Matrix* mpointer, int rowSpot, int colSpot) {
    if (mpointer == NULL || rowSpot <= 0 || rowSpot > mpointer->row || colSpot <= 0 || colSpot > mpointer->col) {
        return 0;
    }
    return 1;
};