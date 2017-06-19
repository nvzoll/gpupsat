#include "CUDAClauseVec.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"

#include <vector>

__host__ CUDAClauseVec::CUDAClauseVec(size_t capacity)
    : capacity { capacity }
{
    check(cudaMalloc(&(this->clauses_dev), capacity * sizeof(Clause)),
          "Creating CUDAClauseVec repository.");
}

__host__ __device__ CUDAClauseVec::~CUDAClauseVec()
{
#ifndef __CUDA_ARCH__
    //cudaFree(clauses_dev);
#endif
}

__host__ __device__ bool CUDAClauseVec::add(Clause& c)
{
    if (size >= capacity) {
        return false;
    }

#ifndef __CUDA_ARCH__
    // If we are on the host, we copy to to device (as the clauses are only on the device)
    cudaMemcpy((clauses_dev + size), &(c), sizeof(Clause), cudaMemcpyHostToDevice);
#else
    // On the device, we simply store the value.
    clauses_dev[size] = c;

    printf("Impossible to add on the host a clause added on the device! (they won't match)\n");

#endif

    size++;
    return true;
}

/**
 * Assuming 0 <= pos < size
 */
__host__ __device__ Clause CUDAClauseVec::get(size_t pos) const
{
#ifndef __CUDA_ARCH__
    Clause *clause = new Clause;
    cudaMemcpy(clause, (clauses_dev + pos), sizeof(Clause), cudaMemcpyDeviceToDevice);
    return *clause;
#else
    return clauses_dev[pos];
#endif
}

__host__ __device__ void CUDAClauseVec::print_all()
{
#ifndef __CUDA_ARCH__
    print_dev <<< 1, 1>>>(*this);
    //cudaDeviceReset();
#else
    printf("This vector contains %d %s:\n", size, size == 1 ? "clause" : "clauses");

    for (int i = 0; i < size; i++) {
        print_clause(clauses_dev[i]);
        printf("\n");
    }
#endif
}

__device__ bool CUDAClauseVec::remove(size_t pos)
{
    if (pos >= size) {
        return false;
    }

    for (int i = pos; i < size - 1; i++) {
        clauses_dev[i] = clauses_dev[i + 1];
    }

    size--;

    return true;
}

__host__ void CUDAClauseVec::alloc_and_copy_to_dev(std::vector<Clause>& vec)
{
#ifdef USE_ASSERTIONS
    assert(vec.size() + size <= capacity);
#endif
    int n_lits = 0;

    for (Clause const& cl : vec) {
        n_lits += cl.n_lits;
    }

    std::vector<Lit> clauses_on_host(n_lits);
    Lit *clauses_on_dev;

    check(cudaMalloc(&clauses_on_dev, sizeof(Lit)*n_lits), "Allocating clauses on device");

    int lit_count = 0;

    for (Clause const& cl : vec) {
        Clause dev_clause;

        dev_clause.capacity = cl.n_lits;
        dev_clause.n_lits = cl.n_lits;
        dev_clause.literals = clauses_on_dev + lit_count;

        bool added = add(dev_clause);

#ifdef USE_ASSERTIONS
        assert(added);
#endif

        for (size_t i = 0; i < cl.n_lits; i++) {
            clauses_on_host[lit_count] = cl.literals[i];
            lit_count++;
        }
    }

    check(cudaMemcpy(clauses_on_dev, clauses_on_host.data(), sizeof(Lit)*n_lits,
                     cudaMemcpyHostToDevice), "copying clauses to device");
}

__global__ void print_dev(CUDAClauseVec vec)
{
    vec.print_all();
}
