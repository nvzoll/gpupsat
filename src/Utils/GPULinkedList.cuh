#ifndef __GPULINKEDLIST_CUH__
#define __GPULINKEDLIST_CUH__

/*
 * Implements a double-linked list in the device. This list stores a copy of the
 * element, so make sure to use a pointer as the template to have the list pointing
 * to a specific element.
 *
 */

#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "SATSolver/Configs.cuh"
#include "NodesRepository.cuh"

template<class T>
class GPULinkedList
{
public:
    struct Node {
        T element;
        Node *next;
        Node *previous;
    };

private:

    Node *first;
    Node *last;
    size_t list_size;
    NodesRepository<Node> *repository;

    __device__ void plain_remove(Node *to_remove)
    {
#ifdef USE_ASSERTIONS
        assert(to_remove != nullptr);
#endif
        Node *previous = to_remove->previous;
        Node *next = to_remove->next;

        if (previous != nullptr) {
            previous->next = next;
        }
        else {
            first = next;
        }

        if (next != nullptr) {
            next->previous = previous;
        }
        else {
            last = previous;
        }

        unallocate_node(to_remove);
        list_size--;

    }

    __device__ Node *get_node_ptr(int index)
    {
#ifdef USE_ASSERTIONS
        assert(index >= 0 && index < list_size);
#endif
        Node *element = first;

        for (int i = 0; i < index; i++) {
            element = element->next;
        }
        return element;

    }
    __device__ Node *allocate_node()
    {
        //printf("Repository = %d\n", repository);
        if (repository == nullptr) {
            Node *node_alloc = new Node;

            //assert(node_alloc != nullptr);
#ifdef USE_ASSERTIONS
            assert(node_alloc != nullptr);
#endif

            return node_alloc;
        }
        else {
            return repository->alloc_element();
        }
    }

    __device__ void unallocate_node(Node *node)
    {
        if (repository == nullptr) {
            delete node;
        }
        else {
            repository->unalloc_element(node);
        }
    }

public:

    __device__ GPULinkedList()
    {
        this->list_size = 0;
        this->first = nullptr;
        this->last = nullptr;
        this->repository = nullptr;
        //assert(false);
    }

    __device__ GPULinkedList(
        NodesRepository<GPULinkedList<T>::Node> *repository)
    {
        this->list_size = 0;
        this->first = nullptr;
        this->last = nullptr;
        this->repository = repository;
        //printf("(1)Repository = %d\n", repository);
    }

    __device__ ~GPULinkedList()
    {
    }


    __device__ size_t size()
    {
        return list_size;
    }

    __device__ bool contains(const T& element)
    {
        LinkedListIterator iter = get_iterator();

        while (iter.has_next()) {
            if (iter.get_next() == element) {
                return true;
            }
        }
        return false;
    }

    __device__ bool remove_obj(const T& element)
    {
        LinkedListIterator iter = get_iterator();

        while (iter.has_next()) {
            if (iter.get_next() == element) {
                iter.remove();
                return true;
            }
        }
        return false;
    }

    __device__ bool empty()
    {
        return size() == 0;
    }

    __device__ void unalloc()
    {
        LinkedListIterator iter = get_iterator();

        while (iter.has_next()) {
            iter.get_next();
            iter.remove();
        }
    }

    /**
     * Returns a pointer to the added element
     */
    __device__ T *add(const T& element, int index)
    {

#ifdef USE_ASSERTIONS
        assert(index >= 0 && index <= list_size);
#endif

        Node *new_node = allocate_node();
        new_node->element = element;

        Node *previous = nullptr;
        Node *next = first;

        for (int i = 0; i < index; i++) {
#ifdef USE_ASSERTIONS
            assert(next != nullptr);
#endif
            previous = next;
            next = next->next;
        }

        if (previous != nullptr) {
            previous->next = new_node;
        }
        else {
            first = new_node;
        }

        if (next != nullptr) {
            next->previous = new_node;
        }
        else {
            last = new_node;
        }

        new_node->next = next;
        new_node->previous = previous;

        list_size++;

        return &(new_node->element);

    }

    /**
     * Make sure an element is present in this list before calling this method!
     */
    __device__ T get(int index)
    {
#ifdef USE_ASSERTIONS
        assert(index >= 0 && index < list_size);
#endif

        Node *element = first;

        for (int i = 0; i < index; i++) {
            element = element->next;
        }

        return element->element;

    }


    __device__ void remove(int index)
    {
#ifdef USE_ASSERTIONS
        assert(index >= 0 && index < list_size);
#endif
        Node *to_remove = first;

        for (int i = 0; i < index; i++) {
            to_remove = to_remove->next;
        }

        plain_remove(to_remove);

    }

    /**
     * Adds an element to the end of the list.
     * Returns a pointer to the added element.
     */
    __device__ T *push_back(const T& element)
    {
        Node *new_node = allocate_node();
#ifdef USE_ASSERTIONS
        assert(new_node != nullptr);
#endif

        new_node->next = nullptr;
        new_node->element = element;

        if (last == nullptr) {
#ifdef USE_ASSERTIONS
            assert(list_size == 0);
#endif
            new_node->previous = nullptr;
            first = new_node;
        }
        else {
            new_node->previous = last;
            last->next = new_node;
        }
        last = new_node;
        list_size++;
        return &(new_node->element);
    }
    __device__ void clear()
    {
        LinkedListIterator iter = get_iterator();

        while (iter.has_next()) {
            iter.get_next();
            iter.remove();
        }
    }

    /**
     * Adds an element to the beginning of the list.
     * Returns a pointer to the added element.
    */
    __device__ T *add_first(const T& element)
    {
        return add(element, 0);
    }


    class LinkedListIterator
    {
    private:
        //Node * current;
        Node *last_sent;
        GPULinkedList *list;
        bool started;

    public:
        __device__ LinkedListIterator(GPULinkedList *the_list, int starting_pos)
        {
            list = the_list;
            if (starting_pos == 0) {
                last_sent = nullptr;
                started = false;
            }
            else {
                last_sent = list->get_node_ptr(starting_pos - 1);
                started = true;
            }
        }
        __device__ LinkedListIterator(GPULinkedList *the_list)
        {
            last_sent = nullptr;
            //current = the_list->first;
            list = the_list;
            started = false;

        }
        __device__ bool has_next()
        {
            if (!started) {
                return list->size() > 0;
            }

            return last_sent->next != nullptr;
            //return current != nullptr;
        }
        __device__ T get_next()
        {

            if (last_sent == nullptr) {
                last_sent = list->first;
                started = true;
            }
            else {
                last_sent = last_sent->next;
            }
            //last_sent = current;
            //current = current->next;
            return last_sent->element;
        }

        __device__ T *get_next_ptr()
        {
            started = true;
            if (last_sent == nullptr) {
                last_sent = list->first;
                started = true;
            }
            else {
                last_sent = last_sent->next;
            }
            //last_sent = current;
            //current = current->next;
            return &(last_sent->element);
        }

        __device__ void remove()
        {
#ifdef USE_ASSERTIONS
            assert(last_sent != nullptr);
#endif
            Node *the_previous = last_sent->previous;
            list->plain_remove(last_sent);

            if (the_previous == nullptr) {
                last_sent = nullptr;
                started = false;
            }
            else {
                last_sent = the_previous;
            }

        }

    };

    __device__ LinkedListIterator get_iterator()
    {
        LinkedListIterator iter(this);
        return iter;
    }

    __device__ LinkedListIterator get_iterator(int starting_position)
    {
        LinkedListIterator iter(this, starting_position);
        return iter;
    }

    __device__ bool check_consistency()
    {
        if (list_size == 0) {
            if (first != nullptr || last != nullptr) {
                printf("No elements but either first or last is not nullptr.\n");
                return false;
            }
            return true;
        }
        else {
            if (first == nullptr || last == nullptr) {
                printf("Not empty but either first or last is nullptr.\n");
                return false;
            }
        }

        Node *node = first;
        Node *previous = nullptr;

        while (node->next != nullptr) {
            if (node->previous != previous) {
                printf("Previous node pointer is not equal to previous!\n");
                return false;
            }
            previous = node;
            node = node->next;
        }

        if (node != last) {
            printf("Last found node is not the last!\n");
            return false;
        }

        return true;
    }

};

#endif /* __GPULINKEDLIST_CUH__ */
