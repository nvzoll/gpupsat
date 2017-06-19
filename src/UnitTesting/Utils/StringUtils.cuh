#ifndef STRINGUTILS_CUH
#define STRINGUTILS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "UnitTesting/TestConfigs.cuh"

__host__ __device__ char *CUDA_strcpy(char *destination, const char *source);

#endif
