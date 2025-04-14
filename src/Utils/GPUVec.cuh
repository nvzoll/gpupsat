#ifndef __GPUVEC_CUH__
#define __GPUVEC_CUH__

#include "SATSolver/SolverTypes.cuh"
#include <assert.h>
#include <memory.h>
#include <string>

template<class T>
class GPUVec
{
private:
    size_t capacity;
    size_t size;
    T *elements;

public:
    __host__ __device__
    GPUVec(size_t capacity)
        : capacity { capacity }
        , size { 0 }
#ifdef __CUDA_ARCH__
        , elements { new T[capacity] }
#endif
    {
#ifndef __CUDA_ARCH__
        cudaMalloc(&elements, capacity * sizeof(T));
#endif
    }

    /**
     * To use an already allocated vector to store the elements!
     */
    __host__ __device__
    GPUVec(T *pointer, size_t capacity, size_t size)
        : capacity { capacity }
        , size { size }
        , elements { pointer }
    {

    }

   __host__ __device__
   void destroy()
   {
#ifndef __CUDA_ARCH__
       cudaFree(elements);
#else
       delete [] elements;
#endif
   }

    __device__ size_t get_capacity() const
    {
        return capacity;
    }

    __host__ __device__ size_t size_of() const
    {
        return size;
    }

    __host__ __device__ bool add(const T& c)
    {
        if (size >= capacity) {
            return false;
        }

#ifndef __CUDA_ARCH__
        // If we are on the host, we copy to the device (as the clauses are only on the device)
        cudaMemcpy((elements + size), &(c), sizeof(T), cudaMemcpyHostToDevice);
        size++;
#else
        // On the device, we simply store the value.
        elements[size] = c;
        size++;
#endif

        return true;
    }

    __host__ __device__ bool remove(size_t pos)
    {
        if (pos >= size) {
            return false;
        }

#ifndef __CUDA_ARCH__
        printf("This is not implemented yet!\n");
        assert(false);
        size--;
#else
        for (size_t i = pos; i < size - 1; i++) {
            elements[i] = elements[i + 1];
        }
        //atomicDec(&size, capacity);
        size--;
#endif
        return true;
    }

    __device__ bool remove_obj(const T& element)
    {
        for (size_t i = 0; i < size; i++) {
            // TODO will this always work?
            if (elements[i] == element) {
                remove(i);
                return true;
            }
        }

        return false;
    }

    __host__ __device__ T get(size_t pos)
    {
#ifdef USE_ASSERTIONS
        assert(pos < size);
#endif

#ifndef __CUDA_ARCH__
        T *element = new T;
        cudaMemcpy(element, (elements + pos), sizeof(T), cudaMemcpyDeviceToHost);
        return *element;
#else
        return elements[pos];
#endif
    }

    __host__ __device__ T *get_ptr(size_t pos) const
    {
#ifdef USE_ASSERTIONS
        assert(pos < size);
#endif

#ifndef __CUDA_ARCH__
        T *element = new T;
        cudaMemcpy(element, (elements + pos), sizeof(T), cudaMemcpyDeviceToHost);
        return element;
#else
        return &elements[pos];
#endif
    }

    /**
     * Resets the value of a position that already contained the element.
     * element: the element to add.
     * pos: the position
     * Return: the old value.
     */
    __host__ __device__ T reset(T element, size_t pos)
    {
#ifdef USE_ASSERTIONS
        assert(pos < size);
#endif
        T old = elements[pos];
        elements[pos] = element;

        return old;
    }

    __host__ __device__ void clear()
    {
        size = 0;
    }

    __host__ __device__ bool empty()
    {
        return size == 0;
    }
    __host__ __device__ bool full()
    {
        return size == capacity;
    }

    __device__ bool contains(const T& element) const
    {
        for (size_t i = 0; i < size; i++) {
            if (elements[i] == element) {
                return true;
            }
        }

        return false;
    }

    __host__ T *copyToHost()
    {
        T *destination;

        destination = new T[size];
        check(cudaMemcpy(destination, elements, sizeof(T)*size, cudaMemcpyDeviceToHost), "Copying to host (GPUVec)");

        return destination;
    }

    __host__ __device__ T last()
    {
        return get(size - 1);
    }

    /**
     * Remove the n last elements of this list (or as many as possible, if n > size)
     */
    __host__ __device__ void remove_n_last(size_t n)
    {
        if (size >= n) {
#ifndef __CUDA_ARCH__
            size -= n;
#else
            //atomicSub(&size, n);
            size -= n;
#endif
        }
        else {
            size = 0;
        }
    }

    __host__ __device__ void copy_to(GPUVec<T> vec)
    {
        for (size_t i = 0; i < size; i++) {
            vec.add(elements[i]);
        }
    }
};


#endif /* __GPUVEC_CUH__ */
