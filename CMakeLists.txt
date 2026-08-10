cmake_minimum_required(VERSION 3.18)
project(NavSolver LANGUAGES CXX)

# C++20 for the FieldStorage concept (src/common/FieldStorage.hpp).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shared layers, by role. Every target gets all three on its include path;
# the backend directory added per-target is what makes SimState resolve
# differently in each binary (see src/core/SimState.hpp).
set(COREDIR   ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(SOLVERDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/solver)
set(IODIR     ${CMAKE_CURRENT_SOURCE_DIR}/src/io)
set(SHARED_INC ${COREDIR} ${SOLVERDIR} ${IODIR})

set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/serial)

file(GLOB SRCS "${SRCDIR}/*.cpp")

# Backend-independent sources, split by whether they define main().
#   CORE_SRCS   geometry + initial conditions — linked by EVERY target,
#               including the test binaries.
#   DRIVER_SRC  entry point + time-marching loops — linked by the CPU solver
#               binaries only. The test binaries bring their own main(), and
#               the CUDA binary has its own driver (src/cuda/main.cu).
# Per-kernel profiling. Off by default and zero-cost when off; see
# src/common/KernelTimers.hpp.
option(NAVSOLVER_PROFILE "Attribute wall time to each per-step kernel" OFF)

if(NAVSOLVER_PROFILE)
    add_compile_definitions(NAVSOLVER_PROFILE)
    message(STATUS "Per-kernel profiling ENABLED — total runtime is not "
                   "comparable to a normal build (see src/common/KernelTimers.hpp)")
endif()

set(CORE_SRCS  ${SOLVERDIR}/Setup.cpp)
set(DRIVER_SRC ${SOLVERDIR}/Driver.cpp)

add_executable(navsolver ${SRCS} ${CORE_SRCS} ${DRIVER_SRC})

# ${SRCDIR} first-class on the include path: SimState.hpp includes
# "BackendConfig.hpp", which must resolve to THIS backend's copy.
target_include_directories(navsolver PRIVATE ${SHARED_INC} ${SRCDIR})

target_compile_options(navsolver PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
)

# Release flags (includes NDEBUG to disable bounds checking)
target_compile_options(navsolver PRIVATE
    $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -march=native -funroll-loops>
)

# Debug flags (no NDEBUG - bounds checking ACTIVE)
target_compile_options(navsolver PRIVATE
    $<$<CONFIG:DEBUG>:-O0 -g3 -DDEBUG -fsanitize=address,undefined -fno-omit-frame-pointer>
)

target_link_options(navsolver PRIVATE
    $<$<CONFIG:DEBUG>:-fsanitize=address,undefined>
)

# Make run target
add_custom_target(run
    COMMAND ./navsolver
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS navsolver
)

# Smoke test: 10 steps on a tiny grid, just to prove the binary runs.
# The config is written at configure time -- inlining it as a shell printf
# inside add_custom_target puts raw newlines into the generated makefile.
set(SMOKE_CFG ${CMAKE_CURRENT_BINARY_DIR}/navsolver_check.cfg)
file(WRITE ${SMOKE_CFG}
"numCellsX = 12
numCellsY = 8
numCellsZ = 8
geometryShape = AbruptExpansion
maxTimeSteps = 10
reportEveryN = 5
outputDir = ${CMAKE_CURRENT_BINARY_DIR}/smoke
")

add_custom_target(check
    COMMAND ${CMAKE_COMMAND} -E echo "Running smoke test..."
    COMMAND $<TARGET_FILE:navsolver> ${SMOKE_CFG}
    COMMAND ${CMAKE_COMMAND} -E echo "Smoke test passed."
    DEPENDS navsolver
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)

# ── OpenMP variant (additive: configure still succeeds without OpenMP) ────────
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    set(OMPSRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/openmp)
    file(GLOB OMPSRCS "${OMPSRCDIR}/*.cpp")

    add_executable(navsolver_omp ${OMPSRCS} ${CORE_SRCS} ${DRIVER_SRC})

    target_include_directories(navsolver_omp PRIVATE ${SHARED_INC} ${OMPSRCDIR})

    target_compile_options(navsolver_omp PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
        -O3 -DNDEBUG -march=native -funroll-loops
    )

    target_link_libraries(navsolver_omp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found — skipping navsolver_omp target")
endif()

# ── CUDA variant (additive — configure still succeeds without a CUDA
#    toolchain). Needs enable_language(CUDA) + mixed C++/CUDA linking, which
#    CMake handles natively; having to reimplement that by hand is what
#    retired the parallel Makefile build. ─────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(CUDASRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cuda)
    file(GLOB CUDASRCS "${CUDASRCDIR}/*.cu")

    # Own driver (main.cu), so no DRIVER_SRC.
    add_executable(navsolver_cuda ${CUDASRCS} ${CORE_SRCS})

    target_include_directories(navsolver_cuda PRIVATE ${SHARED_INC} ${CUDASRCDIR})

    # sm_86: this machine's GPU specifically (NVIDIA GeForce MX570 A,
    # Compute Capability 8.6) -- deliberately narrow, not a generic
    # multi-arch build. See docs/cuda-port.md.
    set_target_properties(navsolver_cuda PROPERTIES
        CUDA_ARCHITECTURES 86
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_compile_options(navsolver_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter -O3 -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
    )

    message(STATUS "CUDA compiler found (${CMAKE_CUDA_COMPILER}) — building navsolver_cuda for sm_86")
else()
    message(STATUS "No CUDA compiler found — skipping navsolver_cuda target")
endif()

# ── Unit tests ───────────────────────────────────────────────────────────────
enable_testing()

file(GLOB_RECURSE TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

add_executable(navsolver_tests ${TEST_SRCS} ${SRCDIR}/Physics.cpp ${CORE_SRCS})

target_include_directories(navsolver_tests PRIVATE
    ${SHARED_INC}
    ${SRCDIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_compile_options(navsolver_tests PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
)

target_compile_definitions(navsolver_tests PRIVATE
    NAVSOLVER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

add_test(NAME unit_tests COMMAND navsolver_tests)

# ── Same test suite, linked against the OpenMP backend ───────────────────────
# The suite only exercises initSimulation() and computeAccelerations(), which
# are supposed to be numerically identical across backends (unlike
# solvePressurePoisson, where OpenMP's red-black SOR is a deliberately
# different algorithm -- see docs/openmp-parallelization.md, and
# scripts/validate_parallel.py which covers that separately). So the golden
# field applies unchanged here: it pins the OpenMP backend to the same
# cell-by-cell accelerations as serial, which nothing in the C++ suite
# checked before.
if(OpenMP_CXX_FOUND)
    add_executable(navsolver_tests_omp ${TEST_SRCS} ${OMPSRCDIR}/Physics.cpp ${CORE_SRCS})

    target_include_directories(navsolver_tests_omp PRIVATE
        ${SHARED_INC}
        ${OMPSRCDIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_compile_options(navsolver_tests_omp PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
    )

    target_compile_definitions(navsolver_tests_omp PRIVATE
        NAVSOLVER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )

    target_link_libraries(navsolver_tests_omp PRIVATE OpenMP::OpenMP_CXX)

    add_test(NAME unit_tests_omp COMMAND navsolver_tests_omp)
endif()

# ── Solver-level validation scripts ──────────────────────────────────────────
# The only checks that exercise a whole built binary end-to-end (VTK field
# comparisons, thread-count equivalence, GPU determinism), so they are the real
# safety net for backend changes. Registered both as ctest tests and as named
# targets, since running one on its own is the common case while iterating.
# NAVSOLVER_BIN_DIR points them at this build tree; --skip-build stops them
# rebuilding what has already been built.
find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    function(add_validation_test name script)
        add_test(NAME ${name}
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${script} --skip-build
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(${name} PROPERTIES
            ENVIRONMENT "NAVSOLVER_BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            TIMEOUT 900
        )

        add_custom_target(${name}
            COMMAND ${CMAKE_COMMAND} -E env
                    NAVSOLVER_BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
                    ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${script} --skip-build
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            USES_TERMINAL)
    endfunction()

    add_validation_test(validate_physics  validate.py)

    if(OpenMP_CXX_FOUND)
        add_validation_test(validate_parallel validate_parallel.py)
    endif()

    if(CMAKE_CUDA_COMPILER)
        add_validation_test(validate_cuda validate_cuda.py)
    endif()

    # Timing harness — not a test (no pass/fail), so target only.
    add_custom_target(bench
        COMMAND ${CMAKE_COMMAND} -E env
                NAVSOLVER_BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark.py --skip-build
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL)
else()
    message(STATUS "Python3 interpreter not found — skipping validation-script tests")
endif()