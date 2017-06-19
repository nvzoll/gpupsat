#ifndef CUDA_LISTGRAPH_CUH_
#define CUDA_LISTGRAPH_CUH_

/*
 * This is an implication graph stored in the GPU. This graphs has literals for vertices,
 * decisions, assumptions and implications are vertices. Vertices that implicate another
 * vertex have a directed edge connecting it to its implication. The backward edge (from
 * implication to the 'implicator') is also stored to allow a backward search.
 *
 */

#include <math.h>
#include <assert.h>
#include <climits>

#include "SATSolver/Configs.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "Utils/GPUVec.cuh"

struct Vertex {
    Decision decision;

#ifdef INCLUDE_FORWARD_EDGES
    int n_neighbors;
#endif

    int n_backward_neighbors;
    bool set;
    bool flagged;
};

struct Edge {
    unsigned int neighbor_index;
    const Clause *implicating_clause;
};

class CUDAListGraph
{

private:
    class GraphStructure
    {
    public:
        // Constructor
        __host__ __device__ GraphStructure(size_t vertices_capacity, size_t ed_capacity);
        __host__ __device__ ~GraphStructure();

        // On vertices
        __device__ void add(Decision d);
        __device__ void remove_vertex(int vertex_index);
        __device__ int get_conflict_vertex_index();

        // On edges
        __device__ void add_edge(int src_vertex_index, int dest_vertex_index, const Clause *new_edge);
        __device__ Edge *get_edge(int src_vertex_index, int edge_index, bool forward);
        __device__ void remove_edge(int vertex_index, int edge_index, bool forward);
        __device__ bool linked(Var src, Var dest);
        __device__ bool linked_with_conflict(Var src);

        // Conflict edge
        __device__ void link_with_conflict(int src_vertex_index,
                                           const Clause *implicating_clause,
                                           int current_decision_level);

        // Other
        __device__ bool contains(Decision decision);
        __device__ void reset();

        // For test
        __device__ bool check_consistency();
        __device__ void print();
        __device__ void print_vertex(Var var);
        __device__ void print_conflict_vertex();


        size_t vertices_capacity;
        size_t edges_capacity;
        Vertex *vertices;

#ifdef INCLUDE_FORWARD_EDGES
        Edge *edges;
#endif
        Edge *backward_edges;
        GPUVec<int> vertices_indices;

    };
    GraphStructure structure;

public:

    class Iterator
    {
    private:
        bool forward;
        int vertex_index;
        int next;
        CUDAListGraph *graph;

    public:
        __device__ Iterator(bool forward,
                            int vertex_index, CUDAListGraph *graph);
        __device__ bool has_next();
        __device__ int get_next_index();
        __device__ bool is_forward();
    };


    __host__ __device__ CUDAListGraph(int vertices_capacity, int ed_capacity);

    __device__ void set(Decision d);
    __device__ bool is_set(Var var);
    __device__ Decision get(int index);
    __device__ void link(Var src, Var dest, const Clause *implicating_clause);
    __device__ int get_n_neighbors(int vertex_index, bool forward);
    __device__ int get_neighbor_index(int vertex_index, int edge_index,
                                      bool forward);
    __device__ int get_conflict_vertex_index();

    /**
     * Removes all vertices whose decision levels are greater or equal to
     * 'decision_level', as well as any edge connecting from or to these vertices.
     */
    __device__ void backtrack_to(int decision_level);
    __device__ void reset();
    __device__ bool contains(Var var);
    __device__ void link_with_conflict(int src_vertex_index,
                                       const Clause *implicating_clause,
                                       int current_decision_level);
    __device__ bool linked(Var src, Var dest);
    __device__ bool linked_with_conflict(Var src);

    __device__ void flag(int vertex_index);
    __device__ void unflag(int vertex_index);
    __device__ void unflag_all();
    __device__ bool are_all_unflagged();
    __device__ bool is_flagged(int vertex_index);

    // Iterator
    __device__ Iterator get_iterator(int vertex_index, bool forward);
    __device__ Iterator get_conflict_vertex_back_iterator();

    // Test methods
    __device__ void print();
    __device__ bool check_consistency();
    __device__ void print_vertex(Var var);
    __device__ void print_conflict_vertex();


};

#endif /* CUDA_LISTGRAPH_CUH_ */
