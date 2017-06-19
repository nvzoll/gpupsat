#ifndef __GEOMETRICRESTARTSMANAGER_CUH__
#define __GEOMETRICRESTARTSMANAGER_CUH__
#include "RestartsManager.cuh"
#include <stdio.h>

class GeometricRestartsManager : RestartsManager
{
private:
    int conflicts_until_restart;
    int n_current_conflicts;
    float increase_factor;
    __device__ void handle_restart();
public:
    __device__ GeometricRestartsManager(int initial_conflicts_until_restart, float increase_factor);
    __device__ void signal_conflict();
    //__device__ ~GeometricRestartManager(){}
    __device__ bool should_restart();
};

#endif /* __GEOMETRICRESTARTSMANAGER_CUH__ */
