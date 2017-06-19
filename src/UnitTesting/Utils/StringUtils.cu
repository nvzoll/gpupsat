#include "StringUtils.cuh"

__host__ __device__ char *CUDA_strcpy(char *destination, const char *source)
{
    int max = MAX_TESTER_NAME - 1;

    int count = 0;

    char c;

    do {
        c = source[count];
        destination[count] = c;
        count++;

        if (count >= max) {
            break;
        }

    }
    while (c != '\0');

    return destination;
}
