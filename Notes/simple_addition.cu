#include <stdio.h>
#include <cuda.h>


__global__
void vecAddKernel(float *A_d, float *B_d, float *C_d, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) 
        C_d[i] = A_d[i] + B_d[i];
}


__host__
void vecAdd(float *A_d, float *B_d, float *C_d, int N)
{
    dim3 DimGrid(ceil(N/256.0), 1, 1);
    dim3 DimBlock(256, 1, 1);
    vecAddKernel<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, N);
} 


int main()
{
    int N = 1024;
    int size = N * sizeof(float);

    // Allocate memory on the host
    float *A_h = (float*)malloc(size);
    float *B_h = (float*)malloc(size);
    float *C_h = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        A_h[i] = i;
        B_h[i] = N - i;
    }

    // Allocate memory on the device
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Call the vector addition function
    vecAdd(A_d, B_d, C_d, N);

    // Copy the result back to the host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] + B_h[i]) {
            printf("Error at index %d: Expected %f, got %f\n", i, A_h[i] + B_h[i], C_h[i]);
            break;
        }
    }
    printf("All values are correct!\n");

    // Free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
