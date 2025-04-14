#ifndef __CUDAVEC_H__
#define __CUDAVEC_H__

#include <assert.h>
#include <vector>

#include "SATSolver/SolverTypes.cuh"
#include "SATSolver/Configs.cuh"

class CUDAClauseVec
{

private:
    size_t capacity;
    size_t size = 0;
    Clause *clauses_dev;

public:
    __host__ CUDAClauseVec(size_t capacity);
    __host__ __device__ ~CUDAClauseVec();

    __host__ __device__ int size_of() const { return size; };
    __host__ __device__ bool add(Clause& c);
    __host__ __device__ Clause get(size_t pos) const;
    __host__ __device__ void print_all();

    __device__ bool remove(size_t pos);
    __device__ const Clause *get_ptr(size_t pos) const { return &clauses_dev[pos]; }

    /**
     * Allocate one continuous space in the device to store all clauses in vec and
     * copies them to it, adding their reference to this object.
     */
    __host__ void alloc_and_copy_to_dev(std::vector<Clause>& vec);
};

__global__ void print_dev(CUDAClauseVec vec);

#endif /* __CUDAVEC_H__ */
