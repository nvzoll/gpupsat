#include "GeometricRestartsManager.cuh"

__device__ GeometricRestartsManager::GeometricRestartsManager(
    int initial_conflicts_until_restart, float increase_factor)
{
    conflicts_until_restart = initial_conflicts_until_restart;
    this->increase_factor = increase_factor;
    n_current_conflicts = 0;
}

__device__ void GeometricRestartsManager::signal_conflict()
{
    n_current_conflicts++;
}

__device__ void GeometricRestartsManager::handle_restart()
{
    n_current_conflicts = 0;
    conflicts_until_restart = (int) (conflicts_until_restart * increase_factor);
}

__device__ bool GeometricRestartsManager::should_restart()
{
    if (n_current_conflicts >= conflicts_until_restart) {
        handle_restart();
        return true;
    }
    else {
        return false;
    }
}
