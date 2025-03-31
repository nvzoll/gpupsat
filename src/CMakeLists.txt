# Use a more recent CMake version for better CUDA/VS support
CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

# Set policies to use modern CMake behavior
cmake_policy(SET CMP0074 NEW) # Use <PackageName>_ROOT variables
cmake_policy(SET CMP0146 NEW) # Avoid deprecated FindCUDA module

# Enable C++, CUDA as project languages
PROJECT(gpupsat LANGUAGES CXX CUDA)

# Explicitly find the CUDA toolkit to ensure targets like CUDA::cudart_static are defined
find_package(CUDAToolkit REQUIRED)

# Manually find cudadevrt library needed for separate compilation, as the imported target might not be defined
find_library(CUDA_DEVRT_LIBRARY cudadevrt HINTS ${CUDAToolkit_LIBRARY_DIR})
if(NOT CUDA_DEVRT_LIBRARY)
    message(FATAL_ERROR "Could not find cudadevrt library in ${CUDAToolkit_LIBRARY_DIR}. Separate compilation requires cudadevrt.lib.")
else()
    message(STATUS "Found cudadevrt library for separate compilation: ${CUDA_DEVRT_LIBRARY}")
endif()

# --- CUDA Compute Capability ---
IF (NOT DEFINED CUDA_COMPUTE_CAPABILITY)
    # Default to a common capability if not specified by user
    # Adjust this default (e.g., "89") based on target hardware if known
    SET(CUDA_COMPUTE_CAPABILITY "89") # Example: "86" for Ampere, "75" for Turing, "61" for Pascal
    MESSAGE(STATUS "CUDA compute capability not specified, using default: ${CUDA_COMPUTE_CAPABILITY}")
    MESSAGE(STATUS "Use CMake flag -DCUDA_COMPUTE_CAPABILITY=XX to override.")
ELSE()
    MESSAGE(STATUS "Using specified CUDA compute capability: ${CUDA_COMPUTE_CAPABILITY}")
ENDIF()
# Set the architecture for CUDA compilation
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}) # Use specific architecture below

# --- Boost Dependency ---
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF) # Typically OFF for MSVC static libs, ON for dynamic

# Find the Boost libraries (ensure BOOST_ROOT is set or Boost is in PATH)
find_package(Boost REQUIRED COMPONENTS program_options)
MESSAGE(STATUS "Found Boost include dirs: ${Boost_INCLUDE_DIRS}")
MESSAGE(STATUS "Found Boost libraries: ${Boost_LIBRARIES}") # For debug

# --- Project Configuration ---
# Include current directory for local headers
INCLUDE_DIRECTORIES(.)

# Define source files
SET(SOURCES
    BCPStrategy/ClauseListStructure.cu
    BCPStrategy/WatchedClausesList.cu
    BCPStrategy/WatchedClausesList.cuh
    ClauseLearning/LearntClauseRepository.cu
    ClauseLearning/LearntClauseRepository.cuh
    ClauseLearning/LearntClausesManager.cu
    ClauseLearning/LearntClausesManager.cuh
    ConflictAnalysis/ConflictAnalyzer.cu
    ConflictAnalysis/ConflictAnalyzer.cuh
    ConflictAnalysis/ConflictAnalyzerFullSearchSpace.cu
    ConflictAnalysis/ConflictAnalyzerFullSearchSpace.cuh
    ConflictAnalysis/ConflictAnalyzerWithWatchedLits.cu
    ConflictAnalysis/ConflictAnalyzerWithWatchedLits.cuh
    ConflictAnalysis/CUDAListGraph.cu
    ConflictAnalysis/CUDAListGraph.cuh
    ConflictAnalysis/GraphAnalyzer.cu
    ConflictAnalysis/GraphAnalyzer.cuh
    ConflictAnalysis/GraphStructure.cu
    DecisionStrategy/Heap.cu
    DecisionStrategy/Heap.cuh
    DecisionStrategy/VarIndexedVector.cu
    DecisionStrategy/VSIDS.cu
    DecisionStrategy/VSIDS.cuh
    ErrorHandler/CudaMemoryErrorHandler.cu
    ErrorHandler/CudaMemoryErrorHandler.cuh
    FileManager/CnfReader.cpp
    FileManager/CnfReader.h
    FileManager/FileUtils.cu
    FileManager/FileUtils.cuh
    FileManager/FormulaData.cu
    FileManager/FormulaData.cuh
    FileManager/ParametersManager.cpp
    FileManager/ParametersManager.h
    FileManager/Version.h
    JobsManager/JobChooser.cu
    JobsManager/JobChooser.cuh
    JobsManager/SimpleJobChooser.cu
    JobsManager/SimpleJobChooser.cuh
    JobsManager/VariableChooser.cu
    JobsManager/VariableChooser.cuh
    Preprocessing/RepeatedLiteralsRemover.cu
    Preprocessing/RepeatedLiteralsRemover.cuh
    Preprocessing/UnaryClausesRemover.cu
    Preprocessing/UnaryClausesRemover.cuh
    Restarts/GeometricRestartsManager.cu
    Restarts/GeometricRestartsManager.cuh
    Restarts/RestartsManager.cuh
    SATSolver/Backtracker.cu
    SATSolver/Backtracker.cuh
    SATSolver/Configs.cuh
    SATSolver/DataToDevice.cu
    SATSolver/DataToDevice.cuh
    SATSolver/DecisionMaker.cu
    SATSolver/DecisionMaker.cuh
    SATSolver/JobsQueue.cu
    SATSolver/JobsQueue.cuh
    SATSolver/main.cu
    SATSolver/Parallelizer.cu
    SATSolver/Parallelizer.cuh
    SATSolver/Results.cu
    SATSolver/Results.cuh
    SATSolver/SATSolver.cu
    SATSolver/SATSolver.cuh
    SATSolver/SolverTypes.cu
    SATSolver/SolverTypes.cuh
    SATSolver/VariablesStateHandler.cu
    SATSolver/VariablesStateHandler.cuh
    Statistics/RuntimeStatistics.cu
    Statistics/RuntimeStatistics.cuh
    Statistics/StatisticsCalculator.cuh
    Utils/CUDAClauseVec.cu
    Utils/CUDAClauseVec.cuh
    Utils/GPULinkedList.cuh
    Utils/GPUStaticVec.cuh
    Utils/GPUVec.cuh
    Utils/NodesRepository.cuh
    Utils/Stopwatch.cuh
)

# Set C++ standard (CMake handles this for CUDA too)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Executable Definition ---
add_executable(gpupsat ${SOURCES})

# --- Linking ---
# Link Boost
target_include_directories(gpupsat PRIVATE ${Boost_INCLUDE_DIRS})
# Use the imported target Boost::program_options for linking
target_link_libraries(gpupsat PRIVATE Boost::program_options)

# Link CUDA libraries (modern CMake handles includes automatically)
# CUDA::cudart_static is typically needed for runtime API
# CUDA::cuda_driver might be needed depending on API usage
# CUDA::cudadevrt is needed for separate compilation / device linking
target_link_libraries(gpupsat PRIVATE CUDA::cudart_static CUDA::cuda_driver ${CUDA_DEVRT_LIBRARY})

# --- Compiler Flags ---
# Add flags common to both C++ and CUDA if needed
# target_compile_options(gpupsat PRIVATE ...)

# Add flags specific to C++
# target_compile_options(gpupsat PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>) # Example for GCC/Clang

# Add flags specific to CUDA (NVCC)
# target_compile_options(gpupsat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler "/W4">) # Example for MSVC host compiler warning level

# Add definitions (e.g., for Boost dynamic linking if needed, though static is set above)
# IF (WIN32 AND NOT Boost_USE_STATIC_LIBS)
#     target_compile_definitions(gpupsat PRIVATE BOOST_ALL_DYN_LINK)
# ENDIF()

# Enable separate compilation (Relocatable Device Code - RDC)
set_target_properties(gpupsat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add -G flag for device debugging in Debug builds
target_compile_options(gpupsat PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)

# Optional: Set output directory for executable
# set_target_properties(gpupsat PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- CUDA Test Executable ---
add_executable(cuda_test cuda_test.cu)
target_link_libraries(cuda_test PRIVATE CUDA::cudart_static)
