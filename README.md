# GPUPSAT: CUDA-Accelerated SAT Solver

GPUPSAT is a CUDA-accelerated Boolean Satisfiability (SAT) solver that leverages GPU parallelism to solve SAT problems efficiently. It implements a parallel DPLL algorithm with modern SAT solving techniques.

## Features

- **GPU Acceleration**: Utilizes CUDA for high-performance parallel solving
- **Dual Execution Modes**: Supports both sequential and parallel execution
- **Modern SAT Techniques**:
  - Conflict analysis with watched literals
  - Clause learning
  - VSIDS decision heuristic
  - Geometric restart strategy
- **Dynamic Workload Distribution**: Job-based parallelization with configurable strategies
- **Performance Monitoring**: Built-in statistics collection
- **CNF File Support**: Processes standard CNF format input files

## Requirements

- CUDA-capable GPU (compute capability 6.1+)
- CUDA Toolkit (11.0+ recommended)
- Boost library (for program options and CNF parsing)
- CMake 3.18+
- C++14 compatible compiler

## Building

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Custom CUDA Compute Capability

To target a specific GPU architecture:

```bash
cmake -DCUDA_COMPUTE_CAPABILITY=86 ..  # For Ampere (RTX 30xx)
```

## Usage

```bash
./gpupsat [options] input.cnf
```

### Command Line Options

- `-i, --input`: Input CNF file (required)
- `-b, --blocks`: Number of CUDA blocks to use (default: 1)
- `-t, --threads`: Number of CUDA threads per block (default: 1)
- `-v, --verbosity`: Output verbosity level (0-3)
- `-p, --preprocess`: Enable unary clause preprocessing
- `-l, --log`: Write performance log to autolog.txt
- `-s, --sequential-as-parallel`: Run sequential algorithm in parallel context (for debugging)

### Example

```bash
./gpupsat -i formula.cnf -b 8 -t 32 -v 1
```

## Architecture

GPUPSAT divides the SAT search space into jobs that can be processed in parallel. Each thread explores a subset of the search space defined by variable assignments. The solver uses:

- **Two-Watched Literals**: For efficient Boolean Constraint Propagation
- **Implication Graph**: For conflict analysis and clause learning
- **JobChooser**: Determines how to split the search space among threads
- **VSIDS**: For dynamic variable selection prioritization

## Performance Notes

- Performance varies based on formula characteristics and hardware
- Best results typically achieved with problem-specific thread/block configurations
- Small problems (<3 variables) automatically run in sequential mode
- Use `-v 1` to see runtime statistics

## License

Apache 2.0

2017
