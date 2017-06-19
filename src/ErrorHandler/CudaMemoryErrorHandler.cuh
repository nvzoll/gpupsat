#include <assert.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void check(cudaError return_val, const char *message);
