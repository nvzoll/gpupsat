#ifndef __GPUVEC_CUH__
#define __GPUVEC_CUH__

#include "SATSolver/SolverTypes.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"
#include <assert.h>
#include <memory.h>
#include <string>
#include <vector>

// Non-owning vector view over a pre-allocated buffer. Usable on host and device.
// The owning host-side counterpart is GPUVecStorage<T>.
template<class T>
class GPUVecView
{
private:
    T *elements_ = nullptr;
    size_t capacity_ = 0;
    size_t size_ = 0;

public:
    __host__ __device__ GPUVecView() = default;
    __host__ __device__ GPUVecView(T *elements, size_t capacity, size_t size = 0)
        : elements_{elements}, capacity_{capacity}, size_{size} {}

    __host__ __device__ size_t capacity() const { return capacity_; }
    __host__ __device__ size_t size_of() const { return size_; }
    __host__ __device__ bool empty() const { return size_ == 0; }
    __host__ __device__ bool full() const { return size_ == capacity_; }
    __host__ __device__ T *raw() const { return elements_; }

    __device__ bool add(const T& v)
    {
        if (size_ >= capacity_) return false;
        elements_[size_++] = v;
        return true;
    }

    __device__ T get(size_t pos) const { return elements_[pos]; }
    __device__ T *get_ptr(size_t pos) const { return &elements_[pos]; }
    __device__ T last() const { return elements_[size_ - 1]; }

    __device__ bool remove(size_t pos)
    {
        if (pos >= size_) return false;
        for (size_t i = pos; i < size_ - 1; ++i) elements_[i] = elements_[i + 1];
        size_--;
        return true;
    }

    __device__ bool remove_obj(const T& element)
    {
        for (size_t i = 0; i < size_; ++i) {
            if (elements_[i] == element) { remove(i); return true; }
        }
        return false;
    }

    __device__ bool contains(const T& element) const
    {
        for (size_t i = 0; i < size_; ++i) {
            if (elements_[i] == element) return true;
        }
        return false;
    }

    __device__ void clear() { size_ = 0; }

    __device__ void remove_n_last(size_t n) { size_ = (size_ >= n) ? size_ - n : 0; }

    __device__ T reset(T element, size_t pos)
    {
        T old = elements_[pos];
        elements_[pos] = element;
        return old;
    }

    __device__ void copy_to(GPUVecView<T> dst) const
    {
        for (size_t i = 0; i < size_; ++i) dst.add(elements_[i]);
    }

    __host__ std::vector<T> copy_to_host() const
    {
        std::vector<T> dest(size_);
        if (size_ > 0) {
            check(cudaMemcpy(dest.data(), elements_, size_ * sizeof(T), cudaMemcpyDeviceToHost),
                  "GPUVecView::copy_to_host");
        }
        return dest;
    }
};

// Host-only owner of a cudaMalloc'd buffer. RAII; non-copyable, move-only.
// Use .view() to obtain a non-owning GPUVecView<T> for device or downstream code.
template<class T>
class GPUVecStorage
{
private:
    T *elements_ = nullptr;
    size_t capacity_ = 0;
    size_t size_ = 0;

public:
    GPUVecStorage() = default;
    explicit GPUVecStorage(size_t capacity) : capacity_{capacity}
    {
        if (capacity_ > 0) {
            check(cudaMalloc(&elements_, capacity_ * sizeof(T)), "GPUVecStorage::cudaMalloc");
        }
    }
    GPUVecStorage(const GPUVecStorage&) = delete;
    GPUVecStorage& operator=(const GPUVecStorage&) = delete;
    GPUVecStorage(GPUVecStorage&& o) noexcept
        : elements_{o.elements_}, capacity_{o.capacity_}, size_{o.size_}
    {
        o.elements_ = nullptr; o.capacity_ = 0; o.size_ = 0;
    }
    GPUVecStorage& operator=(GPUVecStorage&& o) noexcept
    {
        if (this != &o) {
            if (elements_) cudaFree(elements_);
            elements_ = o.elements_; capacity_ = o.capacity_; size_ = o.size_;
            o.elements_ = nullptr; o.capacity_ = 0; o.size_ = 0;
        }
        return *this;
    }
    ~GPUVecStorage() { if (elements_) cudaFree(elements_); }

    bool add(const T& value)
    {
        if (size_ >= capacity_) return false;
        check(cudaMemcpy(elements_ + size_, &value, sizeof(T), cudaMemcpyHostToDevice),
              "GPUVecStorage::add");
        size_++;
        return true;
    }

    size_t size_of() const { return size_; }
    size_t capacity() const { return capacity_; }
    T *raw() const { return elements_; }

    GPUVecView<T> view() const { return GPUVecView<T>(elements_, capacity_, size_); }
    operator GPUVecView<T>() const { return view(); }
};

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
