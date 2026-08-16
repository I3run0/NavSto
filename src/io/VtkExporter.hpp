#pragma once
// =============================================================================
//  VtkExporter.hpp — VTK Legacy ASCII snapshots + a convergence-history CSV.
//
//  STRUCTURED_POINTS matching the uniform Cartesian mesh, readable directly by
//  ParaView/VisIt. Fields per snapshot: Velocity (vector), Pressure,
//  MomentumResidual, VelocityMagnitude. Named {runName}_t{step:06d}.vtk.
//  For large grids, switch to VTK XML (.vts) — binary, same interface.
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

        // 17 significant digits round-trips an IEEE double exactly. Anything
        // less silently quantises the data: at %.6f a converged run wrote a
        // MomentumResidual field whose every non-zero entry was the literal
        // convergenceTol, and scripts/validate_parallel.py's 1e-12 determinism
        // check was comparing rounded values rather than the real ones.
        f << std::setprecision(17);

        // Local extents: this file describes the block of cells this process
        // actually holds. A decomposed run writes one such block per rank, and
        // ORIGIN below places each in the global domain -- their union is the
        // full field, which is VTK's standard piece model.
        const int NX = s.cfg.numCellsX + 1;
        const int NY = s.cfg.numCellsY + 1;
        const int NZ = s.cfg.numCellsZ + 1;

        // ── VTK file header ────────────────────────────────────────────────────
        // The time in the description line is cosmetic, so it is formatted in
        // its own stream: manipulators applied to `f` here would be sticky and
        // would silently reformat every field value written below.
        std::ostringstream title;
        title << "NavSolver snapshot t=" << std::fixed << std::setprecision(6)
              << s.simulationTime << " step=" << step;

        f << "# vtk DataFile Version 3.0\n";
        f << title.str() << "\n";
        f << "ASCII\n";

        // ── Grid definition ────────────────────────────────────────────────────
        f << "DATASET STRUCTURED_POINTS\n";
        f << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n";
        f << "ORIGIN "
          << s.cfg.originX * s.cfg.cellSizeX << " "
          << s.cfg.originY * s.cfg.cellSizeY << " "
          << s.cfg.originZ * s.cfg.cellSizeZ << "\n";
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

    /// Append one convergence-history row.
    ///
    /// The handle is kept open for the process lifetime: this runs once per
    /// timestep, and a long transient (maxTimeSteps=200000) would otherwise pay
    /// 200,000 open+stat+close cycles for one file. Safe because navsolver runs
    /// exactly one simulation per process. The first call TRUNCATES -- a re-run
    /// under the same runName replaces its history rather than concatenating a
    /// second one onto it, which left the step column resetting mid-file.
    static void writeConvergenceCSV(const SimState& s) {
        namespace fs = std::filesystem;
        static std::ofstream csvFile;

        if (!csvFile.is_open()) {
            fs::create_directories(s.cfg.outputDir);
            std::string csvPath = (s.cfg.outputDir / (s.cfg.runName + "_convergence.csv")).string();

            csvFile.open(csvPath, std::ios::trunc);
            if (!csvFile) throw std::runtime_error("VtkExporter: cannot open " + csvPath);

            csvFile << "step,time,ResidMax,ResidRMS,DilMax,IntDiv,IntAbsDiv,dt\n";
        }

        csvFile << std::setprecision(17)
          << s.timeStep        << ","
          << s.simulationTime  << ","
          << s.momentumResidMax << ","
          << s.momentumResidRMS << ","
          << s.dilatationMax   << ","
          << s.intDivergence   << ","
          << s.intAbsDivergence << ","
          << s.timeStepSize    << "\n";
        csvFile.flush();  // keep the file readable by external tools mid-run,
                           // without paying for a full reopen/close per step
    }
};
