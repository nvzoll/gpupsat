#include "SolverTypes.cuh"
#include <stdio.h>
#include <assert.h>
#include "Configs.cuh"

__host__ __device__ Lit mkLit(Var var, bool sign)
{
    Lit p;
    p.x = var + var + (int)sign;
    return p;
}

__host__ __device__ Lit  operator ~(Lit p)
{
    Lit q;
    q.x = p.x ^ 1;
    return q;
}
__host__ __device__ Lit  operator ^(Lit p, bool b)
{
    Lit q;
    q.x = p.x ^ (unsigned int)b;
    return q;
}
__host__ __device__ bool sign      (Lit p)
{
    return p.x & 1;
}
__host__ __device__ int  var       (Lit p)
{
    return p.x >> 1;
}

// Mapping Literals to and from compact integers suitable for array indexing:
__host__ __device__ int  toInt     (Var v)
{
    return v;
}
__host__ __device__ int  toInt     (Lit p)
{
    return p.x;
}
__host__ __device__ Lit  toLit     (int i)
{
    Lit p;
    p.x = i;
    return p;
}

__host__ __device__ void print_lit(Lit l)
{
    printf("%sV%d", sign(l) ? "" : "~", var(l));
}

void create_clause_dev_on_host(int capacity, Clause& c)
{
    c.capacity = capacity;
    check(cudaMalloc(&(c.literals), capacity * (sizeof(Lit))), "Creating clause on GPU from host");
    c.n_lits = 0;
}

void create_clause_host(int capacity, Clause& c)
{
    c.capacity = capacity;
    c.literals = new Lit[capacity];
    c.n_lits = 0;
}

__device__ void create_clause_on_dev(int capacity, Clause& c)
{
    c.capacity = capacity;
    c.literals = new Lit[capacity];
    c.n_lits = 0;
}

__host__ __device__ void addLitToDev(Lit l, Clause& c)
{
#ifdef USE_ASSERTIONS
    assert(c.n_lits < c.capacity);
#endif

#ifndef __CUDA_ARCH__
    check(cudaMemcpy((c.literals + c.n_lits), &l, sizeof(Lit), cudaMemcpyHostToDevice), "Memcpy");
#else
    c.literals[c.n_lits] = l;
#endif

    c.n_lits++;
}

void addLitToHost(Lit l, Clause& c)
{
#ifdef USE_ASSERTIONS
    assert(c.n_lits < c.capacity);
#endif

    memcpy((c.literals + c.n_lits), &l, sizeof(Lit));

    c.n_lits++;
}

void remove_literal(Clause& c, size_t pos)
{
#ifdef USE_ASSERTIONS
    assert(pos >= 0 && pos < c.n_lits);
#endif

    for (size_t i = pos; i < c.n_lits - 1; i++) {
        c.literals[i] = c.literals[i + 1];
    }

    c.n_lits--;
}

__host__ __device__ void print_clause(const Clause& c)
{
    for (size_t i = 0; i < c.n_lits; i++) {
        print_lit(c.literals[i]);
        printf(" ");
    }
    //printf("\n");
}

__global__ void print_clause_kernel(Clause c)
{
    print_clause(c);
}

__host__ void print_dev_clause_on_host(Clause& c)
{
    print_clause_kernel <<< 1, 1>>>(c);
    cudaDeviceReset();
}

__host__ __device__ void print_status(sat_status status)
{
    printf("%s", status == sat_status::SAT ? "SAT" : status == sat_status::UNSAT ? "UNSAT" : "UNDEF");
}

__host__ __device__ void print_decision(Decision d)
{
    print_lit(d.literal);

    printf("(%d)%s%s",
        d.decision_level,
        d.branched ? "." : "",
        d.implicated_from_formula ? "," : "");
}
