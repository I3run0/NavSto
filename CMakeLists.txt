cmake_minimum_required(VERSION 3.18)
project(NavSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/serial)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/common)

file(GLOB SRCS "${SRCDIR}/*.cpp")

add_executable(navsolver ${SRCS})

target_include_directories(navsolver PRIVATE ${INCDIR})

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

# Make check target
add_custom_target(check
    COMMAND ${CMAKE_COMMAND} -E echo "Running smoke test..."
    COMMAND sh -c "printf 'numCellsX = 12\nnumCellsY = 8\nnumCellsZ = 8\ngeometryShape = AbruptExpansion\nmaxTimeSteps = 10\nreportEveryN = 5\n' > /tmp/navsolver_check.cfg"
    COMMAND ./navsolver /tmp/navsolver_check.cfg
    DEPENDS navsolver
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# ── OpenMP variant (additive: configure still succeeds without OpenMP) ────────
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    set(OMPSRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/openmp)
    file(GLOB OMPSRCS "${OMPSRCDIR}/*.cpp")

    add_executable(navsolver_omp ${OMPSRCS})

    target_include_directories(navsolver_omp PRIVATE ${INCDIR})

    target_compile_options(navsolver_omp PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
        -O3 -DNDEBUG -march=native -funroll-loops
    )

    target_link_libraries(navsolver_omp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found — skipping navsolver_omp target")
endif()

# ── CUDA variant (additive, CMake-only per docs/cuda-port.md — configure
#    still succeeds without a CUDA toolchain). Deliberately asymmetric with
#    the OpenMP target above: CUDA needs enable_language(CUDA) + mixed
#    C++/CUDA linking, which CMake handles natively; the hand-rolled
#    Makefile would have to reimplement that, so CUDA is CMake-only. ────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(CUDASRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
    file(GLOB CUDASRCS "${CUDASRCDIR}/*.cu")

    # Reuses src/serial/Physics.cpp AS-IS for initSimulation() (host-only
    # geometry/IC setup, not a per-step hot path) and the other CPU
    # implementations it defines under their original (non-Cuda-suffixed)
    # names -- no symbol clash with Physics.cu's *Cuda-suffixed wrappers,
    # and those CPU implementations simply go unused/unlinked-away rather
    # than needing to be re-derived. See src/cuda/main.cu.
    add_executable(navsolver_cuda ${CUDASRCS} ${SRCDIR}/Physics.cpp)

    target_include_directories(navsolver_cuda PRIVATE ${INCDIR} ${CUDASRCDIR})

    # sm_86: this machine's GPU specifically (NVIDIA GeForce MX570 A,
    # Compute Capability 8.6) -- deliberately narrow, not a generic
    # multi-arch build. See docs/cuda-port.md.
    set_target_properties(navsolver_cuda PROPERTIES
        CUDA_ARCHITECTURES 86
        CUDA_STANDARD 17
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

add_executable(navsolver_tests ${TEST_SRCS} ${SRCDIR}/Physics.cpp)

target_include_directories(navsolver_tests PRIVATE
    ${INCDIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_compile_options(navsolver_tests PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
)

target_compile_definitions(navsolver_tests PRIVATE
    NAVSOLVER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

add_test(NAME unit_tests COMMAND navsolver_tests)