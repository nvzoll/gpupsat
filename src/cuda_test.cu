#include <stdio.h>
#include <stdlib.h> // For exit()
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                                                                       \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t err = call;                                                                                \
        if (err != cudaSuccess)                                                                                \
        {                                                                                                      \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));        \
            /* Optional: Add cudaDeviceReset() here if needed for detailed Nsight Compute/Debugger analysis */ \
            /* cudaDeviceReset(); */                                                                           \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    } while (0)

// Simple kernel that writes thread ID * 2 to output array
__global__ void hello_kernel(int *output)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (output != nullptr)
    {
        output[tid] = tid * 2;
    }
    // printf from device can be unreliable, especially on Windows without specific setup.
    // printf("Hello from thread %d\n", tid);
}

int main()
{
    printf("Starting CUDA test...\n");

    int num_threads = 256;
    int num_blocks = 4; // Use a few blocks
    int data_size = num_threads * num_blocks;
    int *d_output = nullptr; // Device memory pointer
    int *h_output = nullptr; // Host memory for verification

    // Allocate memory on the device
    printf("Allocating device memory (%d ints)...\n", data_size);
    CHECK_CUDA(cudaMalloc(&d_output, data_size * sizeof(int)));
    printf("Device allocation successful.\n");

    // Allocate host memory
    h_output = (int *)malloc(data_size * sizeof(int));
    if (h_output == nullptr)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        if (d_output)
            cudaFree(d_output);
        return EXIT_FAILURE;
    }
    printf("Host allocation successful.\n");

    // Launch the kernel
    printf("Launching hello_kernel (%d blocks, %d threads)...\n", num_blocks, num_threads);
    hello_kernel<<<num_blocks, num_threads>>>(d_output);
    // Check for launch configuration errors immediately after launch syntax
    CHECK_CUDA(cudaGetLastError());
    printf("Kernel launch successful (asynchronous).\n");

    // Synchronize device to ensure kernel completion
    printf("Synchronizing device...\n");
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Synchronization successful.\n");

    // Copy results back to host
    printf("Copying results from device to host...\n");
    CHECK_CUDA(cudaMemcpy(h_output, d_output, data_size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Copy successful.\n");

    // Verify results (optional but good practice)
    bool success = true;
    for (int i = 0; i < data_size; ++i)
    {
        if (h_output[i] != i * 2)
        {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n", i, i * 2, h_output[i]);
            success = false;
            break; // Stop on first error
        }
    }
    if (success)
    {
        printf("Verification successful.\n");
    }

    // Free memory
    printf("Freeing device memory...\n");
    if (d_output)
    {
        CHECK_CUDA(cudaFree(d_output));
    }
    printf("Device free successful.\n");

    printf("Freeing host memory...\n");
    if (h_output)
    {
        free(h_output);
    }
    printf("Host free successful.\n");

    printf("CUDA test completed successfully!\n");
    return 0;
}