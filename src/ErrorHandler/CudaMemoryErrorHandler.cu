#include "CudaMemoryErrorHandler.cuh"

void check(cudaError return_val, const char *message)
{
    if (return_val != cudaSuccess) {
        printf("Error on %s, description: %s\n", message, cudaGetErrorString(return_val));
        cudaDeviceReset();
        exit(1);
    }
}
