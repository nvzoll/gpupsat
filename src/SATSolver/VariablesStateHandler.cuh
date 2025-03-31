#ifndef __VARIABLESSTATEHANDLER_CUH__
#define __VARIABLESSTATEHANDLER_CUH__

#include "Configs.cuh"
#include "SolverTypes.cuh"
#include "../Utils/GPUVec.cuh"
#include "../Utils/GPUStaticVec.cuh"
#include "../Utils/GPULinkedList.cuh"

class VariablesStateHandler;

#include "DecisionMaker.cuh"

class VariablesStateHandler
{
private:
    int n_vars;
    int decision_level = 0;

    DecisionMaker *decision_maker;

    GPUVec<Var> free_vars;
    GPUVec<Decision> decisions;
    GPUVec<Decision> implications;
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> *assumptions;
#else
    GPUStaticVec<Lit> *assumptions;
#endif

    sat_status *vars_status;

    __device__ void undo_decision(int index);
    __device__ void undo_implication(int index);
    __device__ void undo_decision_or_implication(int index, GPUVec<Decision> &list);
    __device__ void free_var(Var v);
    __device__ void block_var(Var v);

public:
    __device__ void reset();

    __device__ VariablesStateHandler(int n_vars,
                                     const Var *dead_vars_elements_ptr, // Use raw pointer
                                     size_t dead_vars_size,             // Use size
                                     DecisionMaker *dec_maker);

    __device__ void set_assumptions(
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
        GPUVec<Lit> *assumptions
#else
        GPUStaticVec<Lit> *assumptions
#endif
    );

    __device__ size_t n_decisions();
    __device__ Decision get_decision(size_t index);
    __device__ size_t n_implications();
    __device__ Decision *get_implication(size_t index);
    __device__ size_t n_assumptions();
    __device__ Lit get_assumption(size_t index);

    __device__ bool assumptions_set();

    __device__ Decision get_last_decision();

    __device__ int get_decision_level();
    __device__ void set_decision_level(int decision_level);

    __device__ void new_implication(Decision implication);
    __device__ void add_many_implications(GPULinkedList<found_implication> &
                                              list_of_implications);
    __device__ void new_decision(Decision decision);
    __device__ Decision unmake_decision(int index_in_decisions);

    /**
     * Returns whether a literal is consistent or not with the current state of
     * decisions, implications and assumptions. A literal is consistent if it is not
     * unsatisfied, that is, if it is free or satisfied.
     */
    __device__ bool is_consistent(Lit lit);
    __device__ int n_consistent_literals(Clause *clause);

    /**
     * Backtracks this objects, removing decisions and implications whose
     * decision levels are greater than new_decision_levels and reseting the decision level.
     *
     * Returns a decision to be handled, which is the removed decision that was made
     * in the decision level next to new_decision level.
     */
    __device__ Decision backtrack_to(int new_decision_level);

    /**
     * Checks if the decision is implied or contradictory to the decisions,
     * implications and assumptions(it does not check the FORMULA itself!).
     * Returns:
     *   sat_status::SAT = The decision is implied by the decisions, implications and/or assumptions.
     *   sat_status::UNSAT = The decision is contradictory to them.
     *   sat_status::UNDEF = The decision is neither implied nor contradictory to the assumptions.
     */

    __device__ sat_status literal_status(Lit lit);
    __device__ sat_status literal_status(Lit lit, bool check_assumptions);
    __device__ sat_status clause_status(Clause clause,
                                        Lit *learnt);
    __device__ bool is_unresolved(Clause *clause);
    __device__ Decision last_non_branched_decision();
    __device__ Decision last_non_branched_decision(int limit_decision_level);
    __device__ int get_decision_level_of(Var v);
    __device__ void increment_decision_level();

    __device__ bool is_var_free(Var v);
    __device__ void free_from_decisions(Decision decision);
    __device__ void free_from_implications(Decision implication);

    /**
     * True = There are no more free vars
     * False = There is at least one free var
     */
    __device__ bool no_free_vars();
    __device__ bool no_decisions();
    __device__ bool no_implications();
    __device__ bool no_assumptions();
    __device__ Var first_free_var();
    __device__ Var last_free_var();

    __device__ bool is_set_as_implicated_from_formula(Var var);
    /**
     * When implicating the literal "implicated literal" from the unit clause "implication_clause"
     * through BCP, this method tells whether this literal should be implicated from formula or not.
     */
    __device__ bool should_implicate_from_formula(Clause const &implication_clause, Lit implicated_literal);

    // Test methods
    __device__ void print_decisions();
    __device__ void print_free_vars();
    __device__ void print_implications();
    __device__ void print_assumptions();
    __device__ void print_var_status();
    __device__ void print_all();

    __device__ bool check_consistency();
};

#endif /* __VARIABLESSTATEHANDLER_CUH__ */
