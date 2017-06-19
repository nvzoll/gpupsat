#ifndef __NODESREPOSITORY_CUH__
#define __NODESREPOSITORY_CUH__

#include "GPULinkedList.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"

#include <vector>

template<class T>
class NodesRepository
{
private:
    T *repository;
    int size;
    int *free_slots;

    __device__ int next_free_index()
    {
        int index = -1;

        for (int i = 0; i < size; i++) {
            if (free_slots[i] == 1) {
                int false_val = 0;
                int old = atomicExch((&free_slots[i]), false_val);

                if (old == 1) {
                    return i;
                }
            }
        }
        return index;
    }

    __device__ void unalloc_element(int index)
    {
        int true_val = 1;
        int old = atomicExch((&free_slots[index]), true_val);
        assert(old == 0);
    }

public:
    NodesRepository(int size)
    {
#ifdef USE_CUDA_MALLOC_FOR_NODES

#else
        this->size = size;
        check(cudaMalloc(&repository, size * sizeof(T)), "Allocating data for node repository");
        check(cudaMalloc(&free_slots, size * sizeof(int)), "Allocating data for node repository");

        //printf("(1) Repository = %d\n", repository);

        std::vector<int> temp(size, 1);

        check(cudaMemcpy(free_slots, temp.data(),
            temp.size() * sizeof(int), cudaMemcpyHostToDevice),
            "Copying data for node repository");
#endif
        //printf("(1) Repository = %d\n", repository);
    }

    __device__ T *alloc_element()
    {
#ifdef USE_CUDA_MALLOC_FOR_NODES
        T *element = new T;

        //assert(element != nullptr);
#ifdef USE_ASSERTIONS
        assert(element != nullptr);
#endif

        return element;

#else
        int index = next_free_index();

        // two attempts.
        if (index < 0) {
            index = next_free_index();
        }

        assert(index >= 0);

        return &(repository[index]);
#endif

    }

    __device__ void unalloc_element(T *element)
    {
#ifdef USE_CUDA_MALLOC_FOR_NODES
        delete element;
#else
        for (int i = 0; i < size; i++) {
            if ((&repository[i]) == element) {
                unalloc_element(i);
                return;
            }
        }

        assert(false);
#endif
    }

    //Test methods
    __device__ bool within_boundaries(T *element)
    {
#ifdef USE_CUDA_MALLOC_FOR_NODES
        return element > 0;
#else
        return element >= repository && element < (repository + size);
#endif

    }
};

#endif /* __NODESREPOSITORY_CUH__ */
