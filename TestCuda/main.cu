#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <windows.h>

//#define BLOCK_NUM 32
//#define THREAD_NUM 256
//#define SIZE BLOCK_NUM * THREAD_NUM
//#define AREA SIZE * SIZE

#define TIME_START QueryPerformanceCounter(&start);

#define TIME_END(NAME) QueryPerformanceCounter(&end); \
printf(NAME);                                         \
printf(" takes %fs\n", (float)(end.QuadPart - start.QuadPart) / (float)freq.QuadPart);

//__global__ void matmul_improvedCUDA(const float* p1, const float* p2, float* ans) {
//    const int bid = blockIdx.x;
//    const int tid = threadIdx.x;
//
//    const int row = bid * THREAD_NUM + tid;
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < SIZE; j++) {
//            ans[row * SIZE + i] += p1[row * SIZE + j] * p2[j * SIZE + i];
//        }
//    }
//}
//
//int main() {
//    LARGE_INTEGER freq;
//    LARGE_INTEGER start;
//    LARGE_INTEGER end;
//    QueryPerformanceFrequency(&freq);
//
//
//
//    float* m1  = (float*) malloc(AREA * sizeof(float));
//    float* m2  = (float*) malloc(AREA * sizeof(float));
//    float* ans  = (float*) malloc(AREA * sizeof(float));
//
//    //initialize the test values
//    for (size_t i = 0; i < AREA; i++) {
//        m1[i] = i;
//        m2[i] = i;
//        ans[i] = 0;
//
//    }
//
//    TIME_START
//    float* gm1;
//    cudaMalloc((void **) &gm1, AREA * sizeof(float));
//    float* gm2;
//    cudaMalloc((void **) &gm2, AREA * sizeof(float));
//    float* gAns;
//    cudaMalloc((void **) &gAns, AREA * sizeof(float));
//
//    cudaMemcpy(gm1, m1, AREA * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(gm2, m2, AREA * sizeof(float), cudaMemcpyHostToDevice);
//
//    matmul_improvedCUDA<<<BLOCK_NUM, THREAD_NUM>>>(gm1, gm2, gAns);
//
//    cudaMemcpy(ans, gAns, AREA * sizeof(float), cudaMemcpyDeviceToHost);
//    TIME_END("CUDA")
//}





#define BLOCK_NUM 32
#define THREAD_NUM 256
#define R_SIZE BLOCK_NUM * THREAD_NUM
#define M_SIZE R_SIZE * R_SIZE

__global__ void mat_mul(int *mat1, int *mat2, int *result) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int row = bid * THREAD_NUM + tid;
    for (int c = 0; c < R_SIZE; c++) {
        for (int n = 0; n < R_SIZE; n++) {
            result[row*R_SIZE+c] += mat1[row*R_SIZE+n] * mat2[n*R_SIZE+c];
        }
    }
}

int main(int argc, char *argv[]) {
    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    QueryPerformanceFrequency(&freq);

    int *mat1, *mat2, *result;
    int *g_mat1, *g_mat2, *g_mat_result;

    mat1 = (int*) malloc(M_SIZE * sizeof(int));
    mat2 = (int*) malloc(M_SIZE * sizeof(int));
    result = (int*) malloc(M_SIZE * sizeof(int));

    for (int i = 0; i < M_SIZE; i++) {
        mat1[i] = rand()/1000000;
        mat2[i] = rand()/1000000;
        result[i] = 0;

    }

    TIME_START
    cudaMalloc((void **)&g_mat1, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat2, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat_result, sizeof(int) * M_SIZE);

    cudaMemcpy(g_mat1, mat1, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);

    mat_mul<<<BLOCK_NUM, THREAD_NUM>>>(g_mat1, g_mat2, g_mat_result);

    cudaMemcpy(result, g_mat_result, sizeof(int) * M_SIZE, cudaMemcpyDeviceToHost);
    TIME_END("CUDA")
}