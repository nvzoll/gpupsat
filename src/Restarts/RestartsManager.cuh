#ifndef __RESTARTSMANAGER_CUH__
#define __RESTARTSMANAGER_CUH__

class RestartsManager
{
public:
    __device__ void signal_conflict();
    __device__ bool should_restart();
};

#endif /* __RESTARTSMANAGER_CUH__ */
