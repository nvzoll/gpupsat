CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(gpupsat)

IF (NOT CUDA_COMPUTE_CAPABILITY)
    SET(CUDA_COMPUTE_CAPABILITY "30")
    MESSAGE("Use flag -DCUDA_COMPUTE_CAPABILITY=XX to set CUDA compute capability.")
    MESSAGE("Compute capability was set to minimal required (30).")
ENDIF()

FIND_PACKAGE(CUDA QUIET REQUIRED)
IF (${CUDA_VERSION_MAJOR} LESS 7)
    MESSAGE(FATAL_ERROR
        "CUDA toolkit v7.0 or newer required (found v${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}).")
ENDIF()

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

IF (WIN32)
    ADD_DEFINITIONS("-DBOOST_ALL_DYN_LINK")
ENDIF()

INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
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

SET(CMAKE_CXX_STANDARD 11)

INCLUDE(FindCUDA)

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION ON)

LIST(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets;")
LIST(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY};")

IF (UNIX)
    LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O0 -ggdb3 -Wall")
ENDIF()

CUDA_ADD_EXECUTABLE(gpupsat ${SOURCES})

GET_FILENAME_COMPONENT(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
FIND_LIBRARY(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
TARGET_LINK_LIBRARIES(gpupsat ${Boost_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})
