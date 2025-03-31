#include "ParametersManager.h"

#include <iostream>
#include <memory>
#include <stdio.h>
#include <string>

#include <boost/config.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/detail/config_file.hpp>
#include <boost/program_options/parsers.hpp>

#include "Version.h"

namespace {

    namespace po = boost::program_options;

    struct program_options_t {
        std::string input_file;
        std::string output_file;
        size_t n_threads;
        size_t n_blocks;
        size_t verbosity_level;
        std::string strategy;
        bool sequential_as_parallel;
        bool preprocess_unary_clauses;
        bool write_log;
    };

    program_options_t get_program_options(int argc, char* argv[])
    {
        program_options_t options;

        // default values
        options.input_file = "task.cnf";
        options.output_file = "solution.txt";
        options.n_threads = 32;
        options.n_blocks = 32;
        options.verbosity_level = 1;
        options.strategy = "distributed";
        options.sequential_as_parallel = false;
        options.preprocess_unary_clauses = true;
        options.write_log = false;

        po::options_description description("runsat: ./runsat <file> [options]");

        description.add_options()("help", "Displays this help message")("version", "Displays version number")("input-file,i", po::value<std::string>(), "Input file")("output-file,o", po::value<std::string>(), "Output file")("number-of-threads,t", po::value<int>(), "Number of threads")("number-of-blocks,b", po::value<int>(), "Number of blocks")("verbosity-level,v", po::value<int>(), "Set verbosity level")("strategy,s", po::value<std::string>(),
            "Strategy for generating the jobs: distributed or uniform")("sequential-as-parallel,p",
                "Forces the execution of the parallel strategy, with jobs creation, but running with 1 thread and 1 block")("preprocess-unary-clauses,u", "Turns on the pre-processing of unary clauses")("write-log,l", "Prints file,threads,blocks,ms to autolog.txt");

        po::positional_options_description pos;
        pos.add("input-file", -1);

        po::variables_map vars;

        try {
            po::store(po::command_line_parser(argc, argv).options(description).positional(pos).run(), vars);
            po::notify(vars);
        }
        catch (const po::error& e) {
            std::cerr << e.what() << std::endl;
            exit(0);
        }

        if (vars.count("help") || vars.size() == 0) {
            std::cout << description;
            exit(0);
        }

        if (vars.count("version")) {
            std::cout << VERSION_STRING << "\n";
            exit(0);
        }

        if (vars.count("input-file")) {
            options.input_file = vars["input-file"].as<std::string>();
        }

        if (vars.count("output-file")) {
            options.output_file = vars["output-file"].as<std::string>();
        }

        if (vars.count("number-of-threads")) {
            options.n_threads = vars["number-of-threads"].as<int>();
        }

        if (vars.count("number-of-blocks")) {
            options.n_blocks = vars["number-of-blocks"].as<int>();
        }

        if (vars.count("verbosity-level")) {
            options.verbosity_level = vars["verbosity-level"].as<int>();
        }

        if (vars.count("strategy")) {
            options.strategy = vars["strategy"].as<std::string>();
        }

        if (vars.count("sequential-as-parallel")) {
            options.sequential_as_parallel = true;
        }

        if (vars.count("preprocess-unary-clauses")) {
            options.sequential_as_parallel = true;
        }

        if (vars.count("write-log")) {
            options.write_log = true;
        }

        if (options.input_file.size() == 0) {
            std::cout << description;
        }

        printf("input file:\t\t\t%s\n", options.input_file.c_str());

        return options;
    }

} // anonymouse namespace

ParametersManager::ParametersManager(int argc, char** argv)
    : correct{ false }
    , has_help{ false }
    , n_threads{ 1 }
    , n_blocks{ 1 }
    , unknown_parameter{ '\0' }
    , verbosity_level{ 0 }
    , strategy{ ChoosingStrategy::DISTRIBUTE_JOBS_PER_THREAD }
    , sequential_as_parallel{ false }
    , preprocess_unary_clauses{ true }
    , write_log{ false }
{
    process(argc, argv);
}

void ParametersManager::process(int argc, char** argv)
{
    auto options = get_program_options(argc, argv);

    input_file = options.input_file;
    output_file = options.output_file;
    n_threads = options.n_threads;
    n_blocks = options.n_blocks;
    verbosity_level = options.verbosity_level;
    sequential_as_parallel = options.sequential_as_parallel;
    preprocess_unary_clauses = options.preprocess_unary_clauses;
    write_log = options.write_log;

    if (options.strategy != "distributed" && options.strategy != "uniform") {
        std::cerr << "Strategy must be either distributed or uniform!\n";
        exit(0);
    }

    strategy = (options.strategy == "uniform")
        ? ChoosingStrategy::UNIFORM
        : ChoosingStrategy::DISTRIBUTE_JOBS_PER_THREAD;

    correct = true;

#if CONFLICT_ANALYSIS_STRATEGY == TWO_WATCHED_LITERALS
    if (!preprocess_unary_clauses) {
        printf("Warning: preprocessing unary clauses is OFF, but it is required for "
            "two-watched literals strategy. Turning it ON now...\n");
        preprocess_unary_clauses = true;
    }
#endif

    return;
}

void ParametersManager::force_sequential_configuration()
{
    n_blocks = 1;
    n_threads = 1;
    sequential_as_parallel = false;
}
