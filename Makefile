# ==============================================================================
#  Makefile — NavSolver  (3-D incompressible Navier-Stokes)
#
#  Targets:
#    make             — Release build (fully optimised)
#    make debug       — Debug build   (bounds checks + AddressSanitizer)
#    make sanitize    — UBSan + ASan  (catches UB and memory errors)
#    make clean       — Remove build artefacts
#    make run         — Release build + run with default config
#    make check       — Build + run a minimal smoke test
#    make test        — Build + run the unit test suite (tests/)
#    make bench       — Build + run the timing harness (scripts/benchmark.py)
#    make validate    — Build + run physics correctness checks (scripts/validate.py)
#    make profile     — Build with gprof instrumentation (-pg); see docs/
#    make kprofile    — Build with per-kernel timing (which kernel got faster?)
#    make kprofile-openmp — same, for the OpenMP variant
#    make openmp      — Build the OpenMP variant (navsolver_omp); see docs/
#    make validate-parallel — OpenMP thread-count equivalence + determinism checks
#
#  Requirements:
#    g++ >= 10 (or clang++ >= 12) with C++20 support (concepts —
#    see src/common/FieldStorage.hpp)
# ==============================================================================

# ── Toolchain ──────────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -Wshadow \
            -Wno-unused-parameter

# ── Directories ────────────────────────────────────────────────────────────────
SRCDIR   := src/serial
INCDIR   := src/common
BUILDDIR := build
TARGET   := navsolver

OMPSRCDIR   := src/openmp
OMPBUILDDIR := build/openmp
OMP_TARGET  := navsolver_omp

TESTDIR       := tests
TESTBUILDDIR  := build/tests
TEST_TARGET   := navsolver_tests

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))

OMPSRCS := $(wildcard $(OMPSRCDIR)/*.cpp)
OMPOBJS := $(patsubst $(OMPSRCDIR)/%.cpp, $(OMPBUILDDIR)/%.o, $(OMPSRCS))

# Backend-independent sources (geometry + initial conditions — src/common/Setup.cpp).
# Compiled once per build profile: the serial and OpenMP variants use different
# flags (-fopenmp), so each gets its own object under its own build dir.
# Listed explicitly, not globbed: Driver.cpp defines main(), so the test
# binaries (which have their own) must link CORE_SRCS without it.
CORE_SRCS   := $(INCDIR)/Setup.cpp
DRIVER_SRC  := $(INCDIR)/Driver.cpp
COMMON_SRCS := $(CORE_SRCS) $(DRIVER_SRC)

COMMON_OBJS    := $(patsubst $(INCDIR)/%.cpp, $(BUILDDIR)/common_%.o, $(COMMON_SRCS))
OMPCOMMON_OBJS := $(patsubst $(INCDIR)/%.cpp, $(OMPBUILDDIR)/common_%.o, $(COMMON_SRCS))

TEST_SRCS := $(shell find $(TESTDIR) -name '*.cpp')
TEST_OBJS := $(patsubst $(TESTDIR)/%.cpp, $(TESTBUILDDIR)/%.o, $(TEST_SRCS))

# Tests link the solver's physics kernels directly (not main.cpp, which
# has its own main()) so PhysicsTests.cpp can exercise them in isolation.
TEST_PHYSICS_OBJ := $(TESTBUILDDIR)/serial_Physics.o

# Setup.cpp only (geometry/ICs) — the tests call initSimulation() directly,
# and must NOT link Driver.cpp, which defines its own main().
TEST_COMMON_OBJS := $(patsubst $(INCDIR)/%.cpp, $(TESTBUILDDIR)/common_%.o, $(CORE_SRCS))

# ── Build profiles ─────────────────────────────────────────────────────────────
RELEASE_FLAGS  := -O3 -DNDEBUG -march=native -funroll-loops
DEBUG_FLAGS    := -O0 -g3 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
SANITIZE_FLAGS := -O1 -g -fsanitize=address,undefined,leak \
                  -fno-omit-frame-pointer
# -O2 (not -O3) to keep function boundaries visible in the call graph —
# aggressive inlining at -O3 can hide where time is actually spent.
PROFILE_FLAGS  := -O2 -g -DNDEBUG -pg
# Per-kernel attribution (src/common/KernelTimers.hpp). Keeps -O3 -- unlike
# PROFILE_FLAGS above, this measures whole kernels, not call graphs, so it
# wants the same optimisation level as a production build.
KPROFILE_FLAGS := -O3 -DNDEBUG -march=native -funroll-loops -DNAVSOLVER_PROFILE
OMP_FLAGS      := -O3 -DNDEBUG -march=native -funroll-loops -fopenmp

# Default: Release
EXTRA_FLAGS ?= $(RELEASE_FLAGS)

# ── Default target ─────────────────────────────────────────────────────────────
.PHONY: all debug sanitize clean run check test bench validate profile kprofile kprofile-openmp openmp validate-parallel

all: $(TARGET)

debug:
	$(MAKE) EXTRA_FLAGS="$(DEBUG_FLAGS)" $(TARGET)

sanitize:
	$(MAKE) EXTRA_FLAGS="$(SANITIZE_FLAGS)" $(TARGET)

profile:
	$(MAKE) EXTRA_FLAGS="$(PROFILE_FLAGS)" $(TARGET)

kprofile:
	$(MAKE) EXTRA_FLAGS="$(KPROFILE_FLAGS)" $(TARGET)

kprofile-openmp:
	$(MAKE) OMP_FLAGS="$(KPROFILE_FLAGS) -fopenmp" $(OMP_TARGET)

openmp: $(OMP_TARGET)

# ── Link ───────────────────────────────────────────────────────────────────────
$(TARGET): $(OBJS) $(COMMON_OBJS)
	@echo "  LINK  $@"
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -o $@ $^

$(OMP_TARGET): $(OMPOBJS) $(OMPCOMMON_OBJS)
	@echo "  LINK  $@"
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o $@ $^

# ── Compile ────────────────────────────────────────────────────────────────────
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -I$(INCDIR) -I$(SRCDIR) -c $< -o $@

$(BUILDDIR)/common_%.o: $(INCDIR)/%.cpp | $(BUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -I$(INCDIR) -I$(SRCDIR) -c $< -o $@

$(OMPBUILDDIR)/%.o: $(OMPSRCDIR)/%.cpp | $(OMPBUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -I$(INCDIR) -I$(OMPSRCDIR) -c $< -o $@

$(OMPCOMMON_OBJS): $(OMPBUILDDIR)/common_%.o: $(INCDIR)/%.cpp | $(OMPBUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -I$(INCDIR) -I$(OMPSRCDIR) -c $< -o $@

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(OMPBUILDDIR):
	@mkdir -p $(OMPBUILDDIR)

# ── Convenience targets ────────────────────────────────────────────────────────
run: all
	./$(TARGET)

check: all
	@echo "Running smoke test (10 steps)…"
	@printf "numCellsX = 12\nnumCellsY = 8\nnumCellsZ = 8\ngeometryShape = AbruptExpansion\nmaxTimeSteps = 10\nreportEveryN = 5\n" \
	    > /tmp/navsolver_check.cfg
	./$(TARGET) /tmp/navsolver_check.cfg
	@echo "Smoke test passed."

test: $(TEST_TARGET)
	./$(TEST_TARGET)

bench:
	python3 scripts/benchmark.py

validate:
	python3 scripts/validate.py

validate-parallel:
	python3 scripts/validate_parallel.py

$(TEST_TARGET): $(TEST_OBJS) $(TEST_PHYSICS_OBJ) $(TEST_COMMON_OBJS)
	@echo "  LINK  $@"
	$(CXX) $(CXXFLAGS) -O0 -g -o $@ $^

$(TESTBUILDDIR)/%.o: $(TESTDIR)/%.cpp | $(TESTBUILDDIR)
	@echo "  CXX   $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -O0 -g -I$(INCDIR) -I$(SRCDIR) -I$(TESTDIR) \
	    -DNAVSOLVER_TEST_DATA_DIR=\"$(CURDIR)/$(TESTDIR)\" -c $< -o $@

$(TEST_PHYSICS_OBJ): $(SRCDIR)/Physics.cpp | $(TESTBUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) -O0 -g -I$(INCDIR) -I$(SRCDIR) -c $< -o $@

$(TEST_COMMON_OBJS): $(TESTBUILDDIR)/common_%.o: $(INCDIR)/%.cpp | $(TESTBUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) -O0 -g -I$(INCDIR) -I$(SRCDIR) -c $< -o $@

$(TESTBUILDDIR):
	@mkdir -p $(TESTBUILDDIR)

clean:
	@echo "  CLEAN"
	@rm -rf $(BUILDDIR) $(TARGET) $(TEST_TARGET) $(OMP_TARGET)
	@rm -f navsolver.log /tmp/navsolver_check.cfg

# ── Dependency tracking (auto-generated) ───────────────────────────────────────
-include $(OBJS:.o=.d)

$(BUILDDIR)/%.d: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(SRCDIR) -MM -MP -MT $(BUILDDIR)/$*.o -MF $@ $<
