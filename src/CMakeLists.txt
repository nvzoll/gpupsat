CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

cmake_policy(SET CMP0074 NEW) # Use <PackageName>_ROOT variables
cmake_policy(SET CMP0146 NEW) # Avoid deprecated FindCUDA module
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)  # legacy FindBoost (header-only Boost layout)
endif()

# CUDA archs must be set before project() — it would otherwise pin SM 5.2.
if(DEFINED CUDA_COMPUTE_CAPABILITY)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 86 89)
endif()

PROJECT(gpupsat LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CUDA_COMPUTE_CAPABILITY
   AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0
   AND NOT "90" IN_LIST CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options)

INCLUDE_DIRECTORIES(.)

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

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(gpupsat ${SOURCES})

target_include_directories(gpupsat PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(gpupsat PRIVATE Boost::program_options CUDA::cudart_static CUDA::cuda_driver)

set_target_properties(gpupsat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add -G flag for device debugging in Debug builds
target_compile_options(gpupsat PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)

# Optional: Set output directory for executable
# set_target_properties(gpupsat PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- CUDA Test Executable ---
# add_executable(cuda_test cuda_test.cu)
# target_link_libraries(cuda_test PRIVATE CUDA::cudart_static)
