cmake_minimum_required(VERSION 3.15)
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
    COMMAND sh -c "printf 'numCellsX = 12\nnumCellsY = 8\nnumCellsZ = 8\nmaxTimeSteps = 10\nreportEveryN = 5\n' > /tmp/navsolver_check.cfg"
    COMMAND ./navsolver /tmp/navsolver_check.cfg
    DEPENDS navsolver
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)