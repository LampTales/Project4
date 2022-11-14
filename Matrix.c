#include "Matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include "innerTools.h"

struct Matrix* createMatrix(int row, int col) {
    if (row <= 0 || col <= 0) {
        return NULL;
    }

    float* fpo = (float*)malloc(row * col * sizeof(float));
    for (int i = 0; i < row * col; i++) {
        fpo[i] = 0;
    }
    struct Matrix* mpo = (struct Matrix*)malloc(sizeof(struct Matrix));
    mpo->row = row;
    mpo->col = col;
    mpo->arr = fpo;
    return mpo;
}

struct Matrix* createMatrixWithIni(int row, int col, const float* fpointer) {
    if (row <= 0 || col <= 0 || fpointer == NULL) {
        return NULL;
    }

    float* fpo = (float*)malloc(row * col * sizeof(float));
    for (int i = 0; i < row * col; i++) {
        fpo[i] = fpointer[i];
    }
    struct Matrix* mpo = (struct Matrix*)malloc(sizeof(struct Matrix));
    mpo->row = row;
    mpo->col = col;
    mpo->arr = fpo;
    return mpo;
}

struct Matrix* copyMatrix(const struct Matrix* mpointer) {
    if (isLegalMatrixWithData(mpointer) == 0) {
        return NULL;
    }
    struct Matrix* m = createMatrixWithIni(mpointer->row, mpointer->col, mpointer->arr);
    return m;
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

int setElement(struct Matrix* mpointer, int rowSpot, int colSpot, float value) {
    int errCode = 0;
    if (isLegalMatrix(mpointer) == 0) {
        return errCode + 1;
    } else if (hasSuchSpot(mpointer, rowSpot, colSpot) == 0) {
        return errCode + 2;
    }

    if (mpointer->arr == NULL) {
        float* fpo = (float*)malloc(mpointer->row * mpointer->col * sizeof(float));
        for (int i = 0; i < (mpointer->row) * (mpointer->col); i++) {
            fpo[i] = 0;
        }
        mpointer->arr = fpo;
    }
    mpointer->arr[(rowSpot - 1) * (mpointer->col) + colSpot - 1] = value;
    return errCode;
}

int reAssignAll(struct Matrix* mpointer, const float* fpointer) {
    int errCode = 10;
    if (isLegalMatrix(mpointer) == 0) {
        return errCode + 1;
    } else if (fpointer == NULL) {
        return errCode + 3;
    }

    if (mpointer->arr == NULL) {
        float* fpo = (float*)malloc(mpointer->row * mpointer->col * sizeof(float));
        mpointer->arr = fpo;
    }
    for (int i = 0; i < (mpointer->row) * (mpointer->col); i++) {
        mpointer->arr[i] = fpointer[i];
    }
    return errCode;
}

int getElement(const struct Matrix* mpointer, int rowSpot, int colSpot, float* answer) {
    int errCode = 20;
    if (isLegalMatrix(mpointer) == 0) {
        return errCode + 1;
    } else if (hasSuchSpot(mpointer, rowSpot, colSpot) == 0) {
        return errCode + 2;
    } else if (answer == NULL) {
        return errCode + 3;
    }
    *answer = mpointer->arr[(rowSpot - 1) * (mpointer->col) + colSpot - 1];
    return errCode;
}

int getMinimum(const struct Matrix* mpointer, float* answer) {
    int errCode = 30;
    if (isLegalMatrixWithData(mpointer) == 0) {
        return errCode + 4;
    }
    float ans = mpointer->arr[0];
    for (int i = 1; i < (mpointer->row) * (mpointer->col); i++) {
        if (mpointer->arr[i] < ans) {
            ans = mpointer->arr[i];
        }
    }
    *answer = ans;
    return errCode;
}

int getMaximum(const struct Matrix* mpointer, float* answer) {
    int errCode = 40;
    if (isLegalMatrixWithData(mpointer) == 0) {
        return errCode + 4;
    }
    float ans = mpointer->arr[0];
    for (int i = 1; i < (mpointer->row) * (mpointer->col); i++) {
        if (mpointer->arr[i] > ans) {
            ans = mpointer->arr[i];
        }
    }
    *answer = ans;
    return errCode;
}

int printMatrix(const struct Matrix* mpointer) {
    int errCode = 50;
    if (isLegalMatrixWithData(mpointer) == 0) {
        return errCode + 4;
    }
    for (int i = 0; i < (mpointer->row) * (mpointer->col); i++) {
        printf("%f\t", mpointer->arr[i]);
        if ((i + 1) % mpointer->col == 0) {
            printf("\n");
        }
    }
    return errCode;
}

void printError(int errCode) {
    if (errCode % 10 == 0) {
        return;
    }

    printf("Errorcode(%d): ", errCode);
    switch (errCode / 10) {
        case 0:
            printf("Error from call of Matrix::setElement: ");
            break;
        case 1:
            printf("Error from call of Matrix::reAssignAll: ");
            break;
        case 2:
            printf("Error from call of Matrix::getElement: ");
            break;
        case 3:
            printf("Error from call of Matrix::getMinimum: ");
            break;
        case 4:
            printf("Error from call of Matrix::getMaximum: ");
            break;
        case 5:
            printf("Error from call of Matrix::printMatrix: ");
            break;
        case 6:
            printf("Error from call of Matrix::addScalar: ");
            break;
        case 7:
            printf("Error from call of Matrix::minusScalar: ");
            break;
        case 8:
            printf("Error from call of Matrix::mulScalar: ");
            break;
        case 9:
            printf("Error from call of Matrix::addMatrix: ");
            break;
        case 10:
            printf("Error from call of Matrix::subtractMatrix: ");
            break;
        case 11:
            printf("Error from call of Matrix::multiplyMatrix: ");
            break;
        default:
            break;
    }

    switch (errCode % 10) {
        case 1:
            printf("The matrix is illegal!\n");
            break;
        case 2:
            printf("There is no such spot in the matrix!\n");
            break;
        case 3:
            printf("The float pointer given is NULL!\n");
            break;
        case 4:
            printf("The matrix is not a legal matrix with data!\n");
            break;
        case 5:
            printf("The input matrixs are not legal matrixs with data!\n");
            break;
        case 6:
            printf("The input matrixs cannot be calculated in this way!\n");
            break;
        case 7:
            printf("The matrix in the output pointer is not a legal matrix!\n");
            break;
        case 8:
            printf("The output matrix and the input matrix cannot be the same matrix!\n");
            break;
        default:
            break;
    }
}