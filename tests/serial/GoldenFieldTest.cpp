#include "Physics.hpp"
#include "SimState.hpp"
#include "TestFramework.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

// Regression test for the loop-order restructuring of computeAccelerations()
// (docs/serial-optimization-loop-order.md). Unlike the Poiseuille analytical
// check in scripts/validate.py -- which seeds the *exact* steady solution as
// the initial condition, so residuals are already ~0 and a subtle bug in the
// restructuring could pass it undetected -- this captures the acceleration
// field itself, cell-by-cell, on a deliberately NON-equilibrium config
// (AbruptExpansion + InletProfile, not Poiseuille), so it's sensitive to
// errors away from equilibrium that the analytical check can't see.
//
// Golden data was captured from the pre-restructuring implementation (see
// tests/serial/golden/accel_abrupt_expansion_24x12x6.txt) and must match
// every subsequent implementation to a tight relative tolerance -- the
// restructuring is supposed to be numerically a no-op (same math, only
// which loop is innermost changes), so any real difference beyond floating-
// point noise indicates a bug in the restructuring, not an intentional
// change.

namespace {

std::string goldenPath(const std::string& name) {
#ifdef NAVSOLVER_TEST_DATA_DIR
    return std::string(NAVSOLVER_TEST_DATA_DIR) + "/serial/golden/" + name;
#else
    return "tests/serial/golden/" + name;  // fallback: assumes CWD == repo root
#endif
}

} // namespace

TEST_CASE(ComputeAccelerations_MatchesGoldenField_AbruptExpansion) {
    SimState s;
    s.cfg.numCellsX = 24;
    s.cfg.numCellsY = 12;
    s.cfg.numCellsZ = 6;
    s.cfg.geometryShape = GeometryShape::AbruptExpansion;
    s.cfg.geometryType = GeometryType::Axial;
    s.cfg.lateralCondition = LateralBC::SolidWall;
    s.cfg.initialProfile = InitialProfile::InletProfile;
    s.cfg.reynoldsNumber = 100.0;
    s.cfg.hyperViscousStart = 0;
    s.cfg.cellSizeX = s.cfg.domainLengthX / s.cfg.numCellsX;
    s.cfg.cellSizeY = s.cfg.domainLengthY / s.cfg.numCellsY;
    s.cfg.cellSizeZ = s.cfg.domainLengthZ / s.cfg.numCellsZ;
    s.allocateFields();
    initSimulation(s);
    computeAccelerations(s);

    std::ifstream f(goldenPath("accel_abrupt_expansion_24x12x6.txt"));
    REQUIRE(f.is_open());

    int nx, ny, nz;
    f >> nx >> ny >> nz;
    REQUIRE(nx == s.cfg.numCellsX);
    REQUIRE(ny == s.cfg.numCellsY);
    REQUIRE(nz == s.cfg.numCellsZ);

    // Relative tolerance for nonzero values; absolute floor for
    // near-zero values where relative comparison is meaningless.
    const double relTol = 1e-12;
    const double absFloor = 1e-14;
    int mismatches = 0;
    double worstRelErr = 0.0;

    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            for (int k = 0; k <= nz; ++k) {
                double gx, gy, gz;
                f >> gx >> gy >> gz;
                double cx = s.accelX(i, j, k);
                double cy = s.accelY(i, j, k);
                double cz = s.accelZ(i, j, k);

                double vals[3] = {cx, cy, cz};
                double golds[3] = {gx, gy, gz};
                for (int c = 0; c < 3; ++c) {
                    double denom = std::max(std::abs(golds[c]), absFloor);
                    double relErr = std::abs(vals[c] - golds[c]) / denom;
                    if (relErr > relTol) {
                        ++mismatches;
                        worstRelErr = std::max(worstRelErr, relErr);
                    }
                }
            }
        }
    }

    if (mismatches > 0) {
        std::ostringstream oss;
        oss << mismatches << " cell(s) mismatched golden field, worst relative error "
            << worstRelErr;
        throw ::testfw::AssertionFailure(oss.str());
    }
}
