#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"
#include "innerTools.h"

int addScalar(struct Matrix* mpointer, float scalar) {
    int errCode = 60;
    if (isLegalMatrixWithData(mpointer) == 0) {
        return errCode + 4;
    }
    for (int i = 0; i < (mpointer->row) * (mpointer->col); i++) {
        mpointer->arr[i] = mpointer->arr[i] + scalar;
    }
    return errCode;
}

int minusScalar(struct Matrix* mpointer, float scalar) {
    int errCode = 70;
    if (isLegalMatrixWithData(mpointer) == 0) {
        return errCode + 4;
    }
    for (int i = 0; i < (mpointer->row) * (mpointer->col); i++) {
        mpointer->arr[i] = mpointer->arr[i] - scalar;
    }
    return errCode;
}

int mulScalar(struct Matrix* mpointer, float scalar) {
    int errCode = 80;
    if (isLegalMatrixWithData(mpointer) == 0) {
        return errCode + 4;
    }
    for (int i = 0; i < (mpointer->row) * (mpointer->col); i++) {
        mpointer->arr[i] = mpointer->arr[i] * scalar;
    }
    return errCode;
}

int addMatrix(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
    int errCode = 90;
    if (isLegalMatrixWithData(mp1) == 0 || isLegalMatrixWithData(mp2) == 0) {
        return errCode + 5;
    } else if (mp1->row != mp2->row || mp1->col != mp2->col) {
        return errCode + 6;
    } else if (isLegalMatrix(answer) == 0) {
        return errCode + 7;
    } else if (mp1 == answer || mp2 == answer) {
        return errCode + 8;
    }

    if (answer->row != mp1->row || answer->col != mp1->col) {
        answer->row = mp1->row;
        answer->col = mp1->col;
        if (answer->arr != NULL) {
            free(answer->arr);
            answer->arr = NULL;
        }
    }
    if (answer->arr == NULL) {
        float* fpo = (float*)malloc(answer->row * answer->col * sizeof(float));
        answer->arr = fpo;
    }

    for (int i = 0; i < (mp1->row) * (mp2->col); i++) {
        answer->arr[i] = mp1->arr[i] + mp2->arr[i];
    }
    return errCode;
}

int subtractMatrix(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
    int errCode = 100;
    if (isLegalMatrixWithData(mp1) == 0 || isLegalMatrixWithData(mp2) == 0) {
        return errCode + 5;
    } else if (mp1->row != mp2->row || mp1->col != mp2->col) {
        return errCode + 6;
    } else if (isLegalMatrix(answer) == 0) {
        return errCode + 7;
    } else if (mp1 == answer || mp2 == answer) {
        return errCode + 8;
    }

    if (answer->row != mp1->row || answer->col != mp1->col) {
        answer->row = mp1->row;
        answer->col = mp1->col;
        if (answer->arr != NULL) {
            free(answer->arr);
            answer->arr = NULL;
        }
    }
    if (answer->arr == NULL) {
        float* fpo = (float*)malloc(answer->row * answer->col * sizeof(float));
        answer->arr = fpo;
    }

    for (int i = 0; i < (mp1->row) * (mp2->col); i++) {
        answer->arr[i] = mp1->arr[i] - mp2->arr[i];
    }
    return errCode;
}

int multiplyMatrix(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer) {
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

    for (int i = 0; i < (mp1->row) * (mp2->col); i++) {
        answer->arr[i] = 0;
        int aspot = i / mp2->col;
        int bspot = i - aspot * mp2->col;
        for (int x = 0; x < mp1->col; x++) {
            answer->arr[i] += mp1->arr[aspot * mp1->col + x] * mp2->arr[x * mp2->col + bspot];
        }
    }
    return errCode;
}