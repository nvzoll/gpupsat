/*
 * ParametersManager.cuh
 *
 *  Created on: Aug 7, 2013
 *      Author: jaime
 */

#ifndef PARAMETERSMANAGER_CUH_
#define PARAMETERSMANAGER_CUH_

#include "JobsManager/JobChooser.cuh"

class ParametersManager
{
public:
    ParametersManager(int argc, char **argv);
    void force_sequential_configuration();

    bool valid_parameters() const
    {
        return correct;
    }

    const char *get_input_file() const
    {
        return input_file.empty() ? nullptr : input_file.c_str();
    }

    const char *get_output_file() const
    {
        return output_file.empty() ? nullptr : output_file.c_str();
    }

    int get_n_threads() const
    {
        return n_threads;
    }

    int get_n_blocks() const
    {
        return n_blocks;
    }

    bool has_help_parameter() const
    {
        return has_help;
    }

    char get_unknown_parameter()
    {
        return unknown_parameter;
    }

    int get_verbosity_level() const
    {
        return verbosity_level;
    }

    ChoosingStrategy get_choosing_strategy() const
    {
        return strategy;
    }

    bool get_sequential_as_parallel() const
    {
        return sequential_as_parallel;
    }

    bool get_preprocess_unary_clauses() const
    {
        return preprocess_unary_clauses;
    }

    bool get_write_log() const
    {
        return write_log;
    }

private:
    bool correct;
    bool has_help;
    std::string input_file;
    std::string output_file;
    int n_threads;
    int n_blocks;
    char unknown_parameter;
    int verbosity_level;
    ChoosingStrategy strategy;
    bool sequential_as_parallel;
    bool preprocess_unary_clauses;
    bool write_log;

    void process(int argc, char **argv);
};

#endif /* PARAMETERSMANAGER_CUH_ */
