#include "UIPFinder.cuh"

__device__ UIPFinder::UIPFinder(CUDAListGraph *impl_graph, int n_vars)
    : var_in_paths()
{
    this->impl_graph = impl_graph;
    this->n_vars = n_vars;
}

__device__ void UIPFinder::paths_to_decision(Decision target_decision, Var current_var,
        GPUStaticVec<Var, 200> *current_path, bool *first_path
                                            )
{
    if (var(target_decision.literal) == current_var) {
        if (*first_path) {
            *first_path = false;
            current_path->copy_to(var_in_paths);
        }
        else {
            for (int i = 0; i < var_in_paths.size_of(); i++) {
                Var var = var_in_paths.get(i);
                if (!current_path->contains(var)) {
                    var_in_paths.remove(i);
                    i--;
                }
            }
        }
        return;
    }

    current_path->add(current_var);

    CUDAListGraph::Iterator iter = impl_graph->get_iterator(current_var, false);

    while (iter.has_next()) {
        int index = iter.get_next_index();

        Decision d = impl_graph->get(index);

        if (d.decision_level == target_decision.decision_level) {
            paths_to_decision(target_decision, var(d.literal), current_path, first_path);
        }

    }

    current_path->remove(current_path->size_of() - 1);

}

__device__ Var UIPFinder::first_uip(Decision target_decision)
{

    GPUStaticVec<Var, 200> current_path;

    int confl = impl_graph->get_conflict_vertex_index();

    bool first = true;

    paths_to_decision(target_decision, confl, &current_path, &first);

    assert(current_path.size_of() >= 2);

    return current_path.get(current_path.size_of() - 2);

}
