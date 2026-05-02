#pragma once
// =============================================================================
//  VtkExporter.hpp  —  Writes simulation fields in VTK Legacy format.
//
//  INDUSTRIAL STANDARD:  The VTK (Visualization Toolkit) format is the
//  dominant standard for 3-D scientific / CFD data exchange.  Exported files
//  can be opened directly in:
//    • ParaView  (https://www.paraview.org)  — the de-facto CFD post-processor
//    • VisIt     (https://visit-dav.github.io/visit-website)
//    • Tecplot, EnSight, and any VTK-capable tool
//
//  FORMAT CHOICE:  We write VTK Legacy ASCII (.vtk) for maximum portability
//  and human-readability.  For large grids, switch to VTK XML (.vts) which
//  supports binary and compressed output — the interface is identical.
//
//  GRID TYPE:  Structured grid (STRUCTURED_POINTS / ImageData) matching the
//  uniform Cartesian mesh used by the solver.
//
//  FIELDS EXPORTED PER SNAPSHOT:
//    • velocity vector  (velX, velY, velZ) → VECTORS field "Velocity"
//    • pressure scalar                     → SCALARS field "Pressure"
//    • momentum residual magnitude         → SCALARS field "MomentumResidual"
//    • velocity divergence                 → SCALARS field "Divergence"
//
//  NAMING CONVENTION:  {runName}_t{step:06d}.vtk
// =============================================================================

#include "SimState.hpp"
#include "Logger.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>

class VtkExporter {
public:
    /// Write a full snapshot of velocity + pressure + diagnostics.
    /// `step` is the time-step index (used in the filename).
    static void writeSnapshot(const SimState& s, int step) {
        namespace fs = std::filesystem;

        // Create output directory if it doesn't exist
        fs::create_directories(s.cfg.outputDir);

        std::ostringstream fname;
        fname << (s.cfg.outputDir / s.cfg.runName).string()
              << "_t" << std::setfill('0') << std::setw(6) << step << ".vtk";

        std::ofstream f(fname.str());
        if (!f)
            throw std::runtime_error("VtkExporter: cannot open " + fname.str());

        const int NX = s.cfg.numCellsX + 1;
        const int NY = s.cfg.numCellsY + 1;
        const int NZ = s.cfg.numCellsZ + 1;

        // ── VTK file header ────────────────────────────────────────────────────
        f << "# vtk DataFile Version 3.0\n";
        f << "NavSolver snapshot t=" << std::fixed << std::setprecision(6)
          << s.simulationTime << " step=" << step << "\n";
        f << "ASCII\n";

        // ── Grid definition ────────────────────────────────────────────────────
        f << "DATASET STRUCTURED_POINTS\n";
        f << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n";
        f << "ORIGIN 0 0 0\n";
        f << "SPACING "
          << s.cfg.cellSizeX << " " << s.cfg.cellSizeY << " " << s.cfg.cellSizeZ << "\n";

        // ── Point data ─────────────────────────────────────────────────────────
        const long long numPts = static_cast<long long>(NX) * NY * NZ;
        f << "\nPOINT_DATA " << numPts << "\n";

        // ── Velocity vector ────────────────────────────────────────────────────
        f << "\nVECTORS Velocity double\n";
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    f << s.velX(i, j, k) << " "
                      << s.velY(i, j, k) << " "
                      << s.velZ(i, j, k) << "\n";

        // ── Pressure scalar ────────────────────────────────────────────────────
        f << "\nSCALARS Pressure double 1\n";
        f << "LOOKUP_TABLE default\n";
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    f << s.press(i, j, k) << "\n";

        // ── Momentum residual magnitude ────────────────────────────────────────
        f << "\nSCALARS MomentumResidual double 1\n";
        f << "LOOKUP_TABLE default\n";
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    f << s.scratchField(i, j, k) << "\n";

        // ── Velocity magnitude (derived, useful for streamlines) ───────────────
        f << "\nSCALARS VelocityMagnitude double 1\n";
        f << "LOOKUP_TABLE default\n";
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    double u = s.velX(i, j, k);
                    double v = s.velY(i, j, k);
                    double w = s.velZ(i, j, k);
                    f << std::sqrt(u*u + v*v + w*w) << "\n";
                }

        f.flush();
        LOG_INFO("VTK snapshot written → ", fname.str());
    }

    /// Write a simple CSV convergence history line-by-line.
    /// File is opened in append mode so successive time steps accumulate.
    static void writeConvergenceCSV(const SimState& s) {
        namespace fs = std::filesystem;
        fs::create_directories(s.cfg.outputDir);

        std::string csvPath = (s.cfg.outputDir / (s.cfg.runName + "_convergence.csv")).string();
        bool isNew = !fs::exists(csvPath);

        std::ofstream f(csvPath, std::ios::app);
        if (!f) throw std::runtime_error("VtkExporter: cannot open " + csvPath);

        if (isNew) {
            f << "step,time,ResidMax,ResidRMS,DilMax,IntDiv,IntAbsDiv,dt\n";
        }
        f << std::fixed << std::setprecision(8)
          << s.timeStep        << ","
          << s.simulationTime  << ","
          << s.momentumResidMax << ","
          << s.momentumResidRMS << ","
          << s.dilatationMax   << ","
          << s.intDivergence   << ","
          << s.intAbsDivergence << ","
          << s.timeStepSize    << "\n";
    }
};
