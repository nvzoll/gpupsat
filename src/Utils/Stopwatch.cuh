#ifndef __STOPWATCH_CUH__
#define __STOPWATCH_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaStopwatch {
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
    float elapsedTime = 0;

public:
    CudaStopwatch()
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }

    void start()
    {
        cudaEventRecord(m_start, 0);
    }

    float stop()
    {
        cudaEventRecord(m_stop, 0);
        cudaEventSynchronize( m_stop );

        cudaEventElapsedTime(&elapsedTime, m_start, m_stop);


        return elapsedTime;
    }

    void reset()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);

        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);

    }

    void destroy()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }
};

#endif /* __STOPWATCH_CUH__ */
