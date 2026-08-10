#include "Physics.hpp"
#include "SimState.hpp"
#include "TestFramework.hpp"

namespace {

SimState makeState(int nx, int ny, int nz, GeometryShape shape, int baseUnit = 0,
                    LateralBC lateral = LateralBC::SolidWall) {
    SimState s;
    s.cfg.numCellsX = nx;
    s.cfg.numCellsY = ny;
    s.cfg.numCellsZ = nz;
    s.cfg.baseUnit = baseUnit;
    s.cfg.geometryShape = shape;
    s.cfg.lateralCondition = lateral;
    s.cfg.cellSizeX = s.cfg.domainLengthX / nx;
    s.cfg.cellSizeY = s.cfg.domainLengthY / ny;
    s.cfg.cellSizeZ = s.cfg.domainLengthZ / nz;
    s.allocateFields();
    return s;
}

} // namespace

// Regression test for a heap-buffer-overflow that used to fire in
// computeAccelerations(): several 1-D coefficient scratch buffers were
// sized one element too small for the west/south/down boundary writes.
TEST_CASE(ComputeAccelerations_DoesNotOverflowOnSmallGrid) {
    SimState s = makeState(12, 8, 8, GeometryShape::AbruptExpansion);
    initSimulation(s);
    computeAccelerations(s); // would heap-buffer-overflow before the fix
}

// Regression test: RoundedCorner/SharpCorner derive jLow/jHigh index bounds
// directly from baseUnit; an incompatible baseUnit used to silently corrupt
// heap memory instead of failing. It must now throw cleanly.
TEST_CASE(InitSimulation_RejectsIncompatibleBaseUnit) {
    SimState s = makeState(12, 8, 8, GeometryShape::RoundedCorner, /*baseUnit=*/20);
    REQUIRE_THROWS(initSimulation(s));
}

TEST_CASE(InitSimulation_AcceptsCompatibleBaseUnit) {
    SimState s = makeState(60, 20, 10, GeometryShape::RoundedCorner, /*baseUnit=*/8,
                            LateralBC::Periodic);
    initSimulation(s); // must not throw
    computeAccelerations(s);
}

// Regression test for a heap-buffer-overflow in solvePressurePoisson(): the
// Neumann ghost-cell mirroring writes one cell past the top of each axis
// (e.g. press(numCellsX+1, j, k)), which needs a ghost layer that
// SimState::allocateFields() didn't provide.
TEST_CASE(SolvePressurePoisson_DoesNotOverflowGhostCells) {
    SimState s = makeState(12, 8, 8, GeometryShape::AbruptExpansion);
    initSimulation(s);
    computeAccelerations(s);
    s.timeStepSize = 0.1;
    s.cfg.numPressureIter = 2;
    buildPressureSource(s);
    solvePressurePoisson(s); // would heap-buffer-overflow before the fix
}
