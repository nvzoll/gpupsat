/*
 * Most of this file is based on the code from Minisat.
 *
 * ***************************************************************************************[Solver.h]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#ifndef SOLVER_TYPES_CUH
#define SOLVER_TYPES_CUH

#include <stdio.h>
#include "Configs.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"

typedef int Var;
#define var_Undef (-1)

/**
 * A literal
 * It is a var with a sign.
 * If a literal has the var v and
 *   - its sign is true, it is the literal v.
 *   - its sign is false, it is the literal ¬v
 */
struct Lit {
    int     x;

    // Use this as a constructor:
    //    __host__ __device__ friend Lit mkLit(Var var, bool sign = false);

    __host__ __device__ bool operator == (Lit p) const
    {
        return x == p.x;
    }
    __host__ __device__ bool operator != (Lit p) const
    {
        return x != p.x;
    }
    __host__ __device__ bool operator <  (Lit p) const
    {
        return x < p.x;     // '<' makes p, ~p adjacent in the ordering.
    }
};

__host__ __device__ Lit mkLit(Var var, bool sign);

__host__ __device__ Lit  operator ~(Lit p);
__host__ __device__ Lit  operator ^(Lit p, bool b);
__host__ __device__ bool sign      (Lit p);
__host__ __device__ int  var       (Lit p);

// Mapping Literals to and from compact integers suitable for array indexing:
__host__ __device__ int  toInt     (Var v);
__host__ __device__ int  toInt     (Lit p);
__host__ __device__ Lit  toLit     (int i);

__host__ __device__ void print_lit(Lit l);


struct Clause {
    Lit *literals;
    unsigned capacity;
    unsigned n_lits;

    __host__ __device__ bool operator == (Clause clause) const
    {
        return clause.literals == literals;
    }

    __host__ __device__ bool contains(Var v) const
    {
        for (unsigned i = 0; i < n_lits; i++) {
            if (var(literals[i]) == v) {
                return true;
            }
        }
        return false;
    }
};

/**
 * Creates a clause on the device from the host.
 */
void create_clause_dev_on_host(int capacity, Clause& c);

__device__ void create_clause_on_dev(int capacity, Clause& c);

/**
 * Creates a clause on the host from the host.
 */
void create_clause_host(int capacity, Clause& c);

/**
 * Adds a literal to a clause on the device.
 * The clause MUST be on the device (otherwise, a segmentation fault may happen!).
 * The function can be called both from the device and the host.
 */
__host__ __device__ void addLitToDev(Lit l, Clause& c);

/**
 * Adds a literal to a clause on the host.
 */
void addLitToHost(Lit l, Clause& c);

__host__ __device__ void print_clause(const Clause& c);
__host__ void print_dev_clause_on_host(Clause& c);
void remove_literal(Clause& c, size_t pos);

enum class sat_status { SAT, UNSAT, UNDEF };

__host__ __device__ void print_status(sat_status status);

/**
 * This struct defines a decision that can be made at any given moment of the solution
 * process. It is also used to store implications (which work just like decisions).
 * This struct holds two data:
 *    literal: The literal indicates the chosen var and its assignment (if the literal
 *    is positive, the assignment is positive. It is analogous to negative).
 *
 *    decision_level: the decision level in which this decision (or implications) was made.
 *    It is important to point out its range: [1, n_vars]. Since
 *    no decision is done on decision level 0, no decision should have this value for its
 *    decision_level.
 */
struct Decision {
    Lit literal;
    int decision_level;
    union {
        bool branched;
        bool implicated_from_formula;
    };


    __host__ __device__ bool operator == (Decision d)
    {
        return literal == d.literal;
    }

};

__host__ __device__ void print_decision(Decision d);

struct found_implication {
    Decision implication;
    Clause *implicating_clause;
};

#endif
