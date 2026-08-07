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
#
#  Requirements:
#    g++ >= 9  (or clang++ >= 10) with C++17 support
# ==============================================================================

# ── Toolchain ──────────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Wshadow \
            -Wno-unused-parameter

# ── Directories ────────────────────────────────────────────────────────────────
SRCDIR   := src/serial
INCDIR   := src/common
BUILDDIR := build
TARGET   := navsolver

TESTDIR       := tests
TESTBUILDDIR  := build/tests
TEST_TARGET   := navsolver_tests

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))

TEST_SRCS := $(shell find $(TESTDIR) -name '*.cpp')
TEST_OBJS := $(patsubst $(TESTDIR)/%.cpp, $(TESTBUILDDIR)/%.o, $(TEST_SRCS))

# Tests link the solver's physics kernels directly (not main.cpp, which
# has its own main()) so PhysicsTests.cpp can exercise them in isolation.
TEST_PHYSICS_OBJ := $(TESTBUILDDIR)/serial_Physics.o

# ── Build profiles ─────────────────────────────────────────────────────────────
RELEASE_FLAGS  := -O3 -DNDEBUG -march=native -funroll-loops
DEBUG_FLAGS    := -O0 -g3 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
SANITIZE_FLAGS := -O1 -g -fsanitize=address,undefined,leak \
                  -fno-omit-frame-pointer
# -O2 (not -O3) to keep function boundaries visible in the call graph —
# aggressive inlining at -O3 can hide where time is actually spent.
PROFILE_FLAGS  := -O2 -g -DNDEBUG -pg

# Default: Release
EXTRA_FLAGS ?= $(RELEASE_FLAGS)

# ── Default target ─────────────────────────────────────────────────────────────
.PHONY: all debug sanitize clean run check test bench validate profile

all: $(TARGET)

debug:
	$(MAKE) EXTRA_FLAGS="$(DEBUG_FLAGS)" $(TARGET)

sanitize:
	$(MAKE) EXTRA_FLAGS="$(SANITIZE_FLAGS)" $(TARGET)

profile:
	$(MAKE) EXTRA_FLAGS="$(PROFILE_FLAGS)" $(TARGET)

# ── Link ───────────────────────────────────────────────────────────────────────
$(TARGET): $(OBJS)
	@echo "  LINK  $@"
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -o $@ $^

# ── Compile ────────────────────────────────────────────────────────────────────
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

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

$(TEST_TARGET): $(TEST_OBJS) $(TEST_PHYSICS_OBJ)
	@echo "  LINK  $@"
	$(CXX) $(CXXFLAGS) -O0 -g -o $@ $^

$(TESTBUILDDIR)/%.o: $(TESTDIR)/%.cpp | $(TESTBUILDDIR)
	@echo "  CXX   $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -O0 -g -I$(INCDIR) -I$(TESTDIR) -c $< -o $@

$(TEST_PHYSICS_OBJ): $(SRCDIR)/Physics.cpp | $(TESTBUILDDIR)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) -O0 -g -I$(INCDIR) -c $< -o $@

$(TESTBUILDDIR):
	@mkdir -p $(TESTBUILDDIR)

clean:
	@echo "  CLEAN"
	@rm -rf $(BUILDDIR) $(TARGET) $(TEST_TARGET)
	@rm -f navsolver.log /tmp/navsolver_check.cfg

# ── Dependency tracking (auto-generated) ───────────────────────────────────────
-include $(OBJS:.o=.d)

$(BUILDDIR)/%.d: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -MM -MP -MT $(BUILDDIR)/$*.o -MF $@ $<
