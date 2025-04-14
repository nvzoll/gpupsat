#ifndef CONFIGS_CUH
#define CONFIGS_CUH

/**
* Debug configuration:
*/
// #define DEBUG
// #define DEBUG_THROUGH_VARIABLES_STATUS
// #define IMPLICATION_GRAPH_DEBUG
#if defined(DEBUG) || defined(IMPLICATION_GRAPH_DEBUG) || defined(DEBUG_THROUGH_VARIABLES_STATUS)
    #define DEBUG_SHOULD_PRINT(THREAD, BLOCK) (THREAD == 0 && BLOCK == 0)
#else
    #define DEBUG_SHOULD_PRINT(THREAD, BLOCK) (0)
#endif
// #define USE_ASSERTIONS
/**
* Solver's configurations:
*/
/**
*  Maximum number of iterations before ending execution (returning sat_status::UNDEF).
*  Do not define this macro to not stop the sat_status::SAT Solver until a solution
*  is found (or it is determined sat_status::UNSAT)
*/
#define MAX_ITERATIONS 1000
#define NULL_DECISION_LEVEL -234
#define USE_CONFLICT_ANALYSIS

#define BASIC_SEARCH 0
#define FULL_SPACE_SEARCH 1
#define TWO_WATCHED_LITERALS 2

// #define CONFLICT_ANALYSIS_STRATEGY BASIC_SEARCH
// #define CONFLICT_ANALYSIS_STRATEGY FULL_SPACE_SEARCH
#define CONFLICT_ANALYSIS_STRATEGY TWO_WATCHED_LITERALS

/**
* Jobs configuration:
*/
#define JOBS_PER_THREAD 10
#define MIN_FREE_VARS 2
#define MAX_VARS 15
#define UNIFORM_NUMBER_OF_VARS 7
#define LARGE_NUMBER INT_MAX
#define MIN_VARIABLES_TO_PARALLELIZE 3
//#define USE_SIMPLE_JOBS_GENERATION


/**
* Data Structure configuration:
*/
// Conflict Graph and learning
#define PROPORTION_OF_NEIGHBOURS_TO_VERTICES 0.25
#define MINIMUM_EDGES 5
//#define INCLUDE_FORWARD_EDGES
#define MAX_NUMBER_OF_NODES 100000
#define MIN_IMPLICATION_PER_VAR 100
#define USE_CUDA_MALLOC_FOR_NODES

/**
* The conflict vertex is treated as a regular vertex by most methods.
* It has a high decision level so it is always removed when a backtracking is carried out.
*/
#define CONFLICT_VERTEX_DECISION_LEVEL INT_MAX
#define MAX_CONFLICTING_VERTICES_SIZE 1000
// Vectors
//#define ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
#define STATIC_GPU_VECTOR_CAPACITY MAX_VARS+1


/**
* Input configuration:
*/
#define MAX_CLAUSE_SIZE 10000

/**
* Watched clauses configuration
*/
//#define MAX_WATCHED_CLAUSES_OBJECTS 100

/**
* Statistics
*/
#define ENABLE_STATISTICS
#define STATISTICS_TIME_OFFSET 7

/**
 * Clause learning configuration
 */
#ifdef USE_CONFLICT_ANALYSIS
    #define USE_CLAUSE_LEARNING
    // #define USE_VSIDS
#endif

#ifdef USE_CLAUSE_LEARNING
    #define MAX_LEARNT_CLAUSES_PER_THREAD 20
#else
    #define MAX_LEARNT_CLAUSES_PER_THREAD 1
#endif


/**
 * Restarts and VSIDS configuration
*/
#define USE_RESTART
#ifdef USE_RESTART
    #define GEOMETRIC_CONFLICTS_BEFORE_RESTART 100
    #define GEOMETRIC_RESTART_INCREASE_FACTOR 1.3
#endif

#define USE_HEAP_IN_VSIDS

#define DEVICE_THREAD_STACK_LIMIT (65536)
#define DEVICE_THREAD_HEAP_LIMIT (1 * 1024 * 1024 * 1024)

#endif // CONFIGS_CUH
