#ifndef __WATCHEDCLAUSESLIST_CUH__
#define __WATCHEDCLAUSESLIST_CUH__

#include <assert.h>
#include <stddef.h>
#include <limits.h>

#include "SATSolver/SolverTypes.cuh"
#include "Utils/GPULinkedList.cuh"
#include "Utils/GPUVec.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "SATSolver/Configs.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "Utils/NodesRepository.cuh"

/*
 * WatchedClausesList.cuh
 *
 * This class implements the two watched literals strategy. It holds the list of
 * clauses for each var which are watching its equivalent var.
 *
 */

struct WatchedClause {
    Clause clause;
    int watched_lit_index_1;
    int watched_lit_index_2;
};

using watched_clause_node_t = NodesRepository<GPULinkedList<WatchedClause *>::Node>;

class WatchedClausesList
{
public:
    __device__ WatchedClausesList(int n_vars,
                                  VariablesStateHandler *handler,
                                  watched_clause_node_t *node_repository,
                                  //GPUVec< WatchedClause > * repository,
                                  int n_clauses);

    /**
     * Adds a new clause to this object.
     * This method must be called before starting solving.
     * When adding learnt clauses, this method must be modified to allow
     * verifying the clause current states and what literals to watch.
     */
    __device__ void new_clause(const Clause *clause, Lit& implicated, bool check_status);
    __device__ void new_clause(const Clause *clause);
    __device__ void add_all_clauses(CUDAClauseVec const& formula);
    __device__ sat_status new_decision(Decision decision,
                                       const Clause **conflicting_clause,
                                       GPULinkedList<found_implication>& implication_list);
    __device__ void replace_clause(Clause c, Clause new_clause);
    /**
     * Handles backtrack.
     */
    __device__ void handle_backtrack();
    __device__ void reset();


    // Test methods
    __device__ bool check_consistency();
    __device__ bool check_clause(Clause *clause, bool& blocked);
    __device__ void print_structure();
    __device__ bool contains(Var v, Clause c);
    __device__ bool blocked_contains(Clause c);

private:

    VariablesStateHandler *vars_handler;
    GPUVec<WatchedClause> *repository;
    int next_watched_clause_index;

    /**
     * Processes a clause, checking which literals are inconsistent and changing
     * them accordingly. If an implication is found, it is added to the implications list,
     * but it is not processed (MUST BE PROCESSED AFTER THAT!)
     *
     * Return: True if a conflict was caused by this clause, false otherwise.
     */
    __device__ bool process_clause(WatchedClause *watch_clause, Decision decision,
                                   GPULinkedList<found_implication>& implication_list);

    /**
     * Add a new clause and return an implicated literal iff the clause is unit,
     * otherwise literal.x == -1.
     * Clause: clause to add
     * watched_clause: the watched clause generated.
     * literal: the unit literal (if any)
     * check_status: check status of clause. If true, checks if the clause is unit
     * or sat_status::UNSAT, blocking it in this case. Otherwise, assumes it is neither unit nor
     * sat_status::UNSAT and watches it.
     */
    __device__ void new_clause(Clause clause,
                               WatchedClause *watched_clause, Lit& literal, bool check_status);

    /**
     * Handles an implications caused by a unary clause.
     * TODO Must not forget to remove the clause from the watched literals (check if it is in the other)
     * and add to a list
     */
    __device__ void handle_implication(WatchedClause *watched_clause, Lit unit_literal,
                                       GPULinkedList<found_implication>& implication_list, Decision original_implication);

    /**
     * Handles a conflict.
     * TODO Must not forget to remove the clause from the watched literals
     */
    __device__ void handle_conflict(WatchedClause *watched_clause);
    __device__ bool is_currently_consistent(Lit literal,
                                            GPULinkedList<found_implication>& partial_implications);
    __device__ sat_status literal_status(Lit literal,
                                         GPULinkedList<found_implication>& implication_list);

    class ClauseListStructure
    {
    private:
        GPULinkedList<WatchedClause *> *watched_clauses_per_var;
        GPULinkedList<WatchedClause *> blocked_clauses;
        VariablesStateHandler *vars_handler;
        int n_vars;
    public:
        int n_added_clauses;
        __device__ ClauseListStructure(int n_vars,
                                       VariablesStateHandler *var_handler);
        __device__ ClauseListStructure(int n_vars,
                                       VariablesStateHandler *var_handler,
                                       watched_clause_node_t *node_repository
                                      );
        __device__ void add_clause(WatchedClause *clause, Var var);
        __device__ bool remove_clause(Clause *clause_ptr, Var var);
        __device__ WatchedClause *get_clause(Var var, int clause_index);
        __device__ void switch_watched_literal(int old_watched_index, int new_watched_index,
                                               WatchedClause *clause);
        __device__ int clauses_list_size(Var var);
        __device__ GPULinkedList<WatchedClause *>::LinkedListIterator get_iterator(Var var);
        __device__ void block_clause(WatchedClause *clause);
        __device__ void unblock_clause(WatchedClause *clause);
        __device__ void unblock_all();
        __device__ void purge_clause(Clause c);
        __device__ bool purge_clause_from_blocked(Clause c);
        /**
         * Watch blocked clauses. Does not check consistency, because it assumes
         * assumptions, decisions and implications will be (or have been) undone.
         */
        __device__ void reset();
        /**
         * Returns the indices of the two literals in clause 'c' with the two smallest
         * lits sizes. If test consistency is set to true, it considers only
         * consistent literals.
         */
        __device__ void get_two_literals_with_fewer_clauses(Clause *c, int& smallest,
                int& second_smallest, bool test_consistency);
        /**
        * Choose 2 consistent literals to watch and adds the current clause to them.
        * If test_consistency == true, it tests the literals to choose consistent ones
        * (otherwise, all MUST be consistent, to avoid watching inconsistent literals).
        */
        __device__ void watch_clause(WatchedClause *watched_clause, bool test_consistency);

        //Test methods
        __device__ bool contains(Var v, Clause c);
        __device__ bool blocked_contains(Clause c);
        __device__ bool check_consistency();
        /**
         * Check whether the clause is present and coherent.
         * Also tells if it is blocked or not.
         */
        __device__ bool check_clause(Clause *clause, bool& blocked);
        __device__ void print_structure();
        __device__ void print_blocked_clauses();
    };

    ClauseListStructure structure;
};

// For tests
__host__ __device__ void print_watch_clause(WatchedClause& clause);

#endif /* __WATCHEDCLAUSESLIST_CUH__ */
