#pragma once

struct Matrix* createRawMatrix(int row, int col);

int isLegalMatrix(const struct Matrix* mpointer);

int isLegalMatrixWithData(const struct Matrix* mpointer);

int hasSuchSpot(const struct Matrix* mpointer, int rowSpot, int colSpot);