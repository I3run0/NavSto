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
#
#  Requirements:
#    g++ >= 9  (or clang++ >= 10) with C++17 support
# ==============================================================================

# ── Toolchain ──────────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Wshadow \
            -Wno-unused-parameter

# ── Directories ────────────────────────────────────────────────────────────────
SRCDIR   := src
INCDIR   := include
BUILDDIR := build
TARGET   := navsolver

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))

# ── Build profiles ─────────────────────────────────────────────────────────────
RELEASE_FLAGS  := -O3 -DNDEBUG -march=native -funroll-loops
DEBUG_FLAGS    := -O0 -g3 -DDEBUG -fsanitize=address,undefined \
                  -fno-omit-frame-pointer
SANITIZE_FLAGS := -O1 -g -fsanitize=address,undefined,leak \
                  -fno-omit-frame-pointer

# Default: Release
EXTRA_FLAGS ?= $(RELEASE_FLAGS)

# ── Default target ─────────────────────────────────────────────────────────────
.PHONY: all debug sanitize clean run check

all: $(TARGET)

debug:
	$(MAKE) EXTRA_FLAGS="$(DEBUG_FLAGS)" $(TARGET)

sanitize:
	$(MAKE) EXTRA_FLAGS="$(SANITIZE_FLAGS)" $(TARGET)

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
	@printf "numCellsX = 12\nnumCellsY = 8\nnumCellsZ = 8\nmaxTimeSteps = 10\nreportEveryN = 5\n" \
	    > /tmp/navsolver_check.cfg
	./$(TARGET) /tmp/navsolver_check.cfg
	@echo "Smoke test passed."

clean:
	@echo "  CLEAN"
	@rm -rf $(BUILDDIR) $(TARGET)
	@rm -f navsolver.log /tmp/navsolver_check.cfg

# ── Dependency tracking (auto-generated) ───────────────────────────────────────
-include $(OBJS:.o=.d)

$(BUILDDIR)/%.d: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -MM -MP -MT $(BUILDDIR)/$*.o -MF $@ $<
