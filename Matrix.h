#pragma once

// A matrix lib that based on float numbers.
struct Matrix {
    int row;
    int col;
    float* arr;
};

float dotproduct_avx2(const float *p1, const float * p2, size_t n);

int matmul_plain(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

int matmul_improved(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

// Create a matrix that all the elements are zeros.
// If row or col is illegal, it will return NULL.
struct Matrix* createMatrix(int row, int col);

// Create a matrix that the elements are setted as the data in the float pointer.
// If the parameters are illegal, it will return NULL.
// Please make sure that the float pointer has enough data for the matrix, otherwise segment errors may occour.
struct Matrix* createMatrixWithIni(int row, int col, const float* fpointer);

// Clone the given matrix and return.
// If the given matrix is illegal, it will return NULL.
struct Matrix* copyMatrix(const struct Matrix* mpointer);

// Please remember to delete the Matrix when you finish using it.
// Input the pointer pointing to the matrix pointer, this function will free it and point the matrix pointer to NULL.
void deleteMatrix(struct Matrix** mpp);

// Set the element at the given spot to the given value.
// Return error code: 0X, use printError function to output error information on the terminal.
int setElement(struct Matrix* mpointer, int rowSpot, int colSpot, float value);

// Set all the elements as the data in the float pointer.
// Return error code: 1X, use printError function to output error information on the terminal.
// Please make sure that the float pointer has enough data for the matrix, otherwise segment errors may occour.
int reAssignAll(struct Matrix* mpointer, const float* fpointer);

// Find the element at the given spot and put it in tothe given float pointer.
// Return error code: 2X, use printError function to output error information on the terminal.
// Please make sure that the float pointer is not wild or null.
int getElement(const struct Matrix* mpointer, int rowSpot, int colSpot, float* answer);

// Get the minumum value in the matrix and put it into the given float pointer.
// Return error code: 3X, use printError function to output error information on the terminal.
// Please make sure that the float pointer is not wild.
int getMinimum(const struct Matrix* mpointer, float* answer);

// Get the maxumum value in the matrix and put it into the given float pointer.
// Return error code: 4X, use printError function to output error information on the terminal.
// Please make sure that the float pointer is not wild.
int getMaximum(const struct Matrix* mpointer, float* answer);

// Print the data in the matrix to the terminal.
// Return error code: 5X, use printError function to output error information on the terminal.
// Please do not apply this function to the matrix structs that are not created by given functions, otherwise segment errors may occour.
int printMatrix(const struct Matrix* mpointer);

// Add a scalar to all the elements in the matrix.
// Return error code: 6X, use printError function to output error information on the terminal.
int addScalar(struct Matrix* mpointer, float scalar);

// Minus a scalar to all the elements in the matrix.
// Return error code: 7X, use printError function to output error information on the terminal.
int minusScalar(struct Matrix* mpointer, float scalar);

// Mul a scalar to all the elements in the matrix.
// Return error code: 8X, use printError function to output error information on the terminal.
int mulScalar(struct Matrix* mpointer, float scalar);

// Do addtion to two matrix and put the answer into the thrid matrix.
// Return error code: 9X, use printError function to output error information on the terminal.
// The function can do adjustments to the answer matrix, but you should make sure that it is not wild.
int addMatrix(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

// Do subtraction to two matrix and put the answer into the thrid matrix.
// Return error code: 10X, use printError function to output error information on the terminal.
// The function can do adjustments to the answer matrix, but you should make sure that it is not wild.
int subtractMatrix(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

// Do multiplication to two matrix and put the answer into the thrid matrix.
// Return error code: 11X, use printError function to output error information on the terminal.
// The function can do adjustments to the answer matrix, but you should make sure that it is not wild.
int multiplyMatrix(const struct Matrix* mp1, const struct Matrix* mp2, struct Matrix* answer);

// According to the error code, it can print out the error information.
void printError(int errCode);
