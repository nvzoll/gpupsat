#ifndef __GPUSTATICVEC_H__
#define __GPUSTATICVEC_H__

/*
 *  A vector that is statically created on GPU, avoiding memory latency issues.
 *
 */

#include "SATSolver/SolverTypes.cuh"
#include "SATSolver/Configs.cuh"
#include <assert.h>

template<class D, int VECTOR_STATIC_SIZE = STATIC_GPU_VECTOR_CAPACITY>
class GPUStaticVec
{
private:
    const unsigned int capacity = VECTOR_STATIC_SIZE;
    unsigned int size;
    D elements[VECTOR_STATIC_SIZE];

public:

    __device__ GPUStaticVec() : size { 0 }
    {

    }

    __device__ ~GPUStaticVec()
    {

    }
    __device__ unsigned int get_capacity()
    {
        return capacity;
    }
    __device__ unsigned int size_of()
    {
        return size;
    }

    __device__ bool add(const D& c)
    {
        if (size >= capacity) {
            return false;
        }

        // On the device, we simply store the value.
        elements[size++] = c;

        return true;
    }

    __device__ bool remove(int pos)
    {
        if (pos < 0 || pos >= size) {
            return false;
        }

        for (int i = pos; i < size - 1; i++) {
            elements[i] = elements[i + 1];
        }
        //atomicDec(&size, capacity);
        size--;
        return true;
    }

    __device__ bool remove_obj(const D& element)
    {
        for (int i = 0; i < size; i++) {
            // TODO will this always work?
            if (elements[i] == element) {
                remove(i);
                return true;
            }
        }

        return false;
    }

    __device__ D get(int pos)
    {
#ifdef USE_ASSERTIONS
        assert(pos >= 0 && pos < size);
#endif

        return elements[pos];
    }

    __device__ D *get_ptr(int pos)
    {
        if (pos >= size) {
            printf("Out of bounds!\n");
            return nullptr;
        }

        return &(elements[pos]);
    }

    __device__ void clear()
    {
        size = 0;
    }

    __device__ bool empty()
    {
        return size == 0;
    }

    __device__ bool contains(const D& element)
    {
        for (int i = 0; i < size; i++) {
            if (elements[i] == element) {
                return true;
            }
        }

        return false;
    }

    __host__ D *copyToHost()
    {
        D *destination = new D[size];

        check(cudaMemcpy(destination, elements, sizeof(D)*size, cudaMemcpyDeviceToHost), "Copying static vector to host");

        return destination;
    }

    __host__ __device__ bool full()
    {
        return size == capacity;
    }

    /**
     * Resets the value of a position that already contained the element.
     * element: the element to add.
     * pos: the position
     * Return: the old value.
    */
    __host__ __device__ D reset(D element, int pos)
    {
#ifdef USE_ASSERTIONS
        assert(pos < size && pos >= 0);
#endif
        D old = elements[pos];
        elements[pos] = element;
        return old;
    }


    /**
     * Remove the n last elements of this list (or as many as possible, if n > size)
     */
    __device__ void remove_n_last(int n)
    {
        if (size >= n) {
            //atomicSub(&size, n);
            size -= n;
        }
        else {
            size = 0;
        }
    }

    template<int THE_SIZE>
    __device__ void copy_to(GPUStaticVec<D, THE_SIZE> vec)
    {
        for (int i = 0; i < size; i++) {
            vec.add(elements[i]);
        }
    }


};


#endif /* __GPUSTATICVEC_H__ */
