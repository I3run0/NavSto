    #pragma once
    // =============================================================================
    //  SimState.hpp  —  Complete simulation state (replaces scattered globals).
    //
    //  Every physical field, grid parameter, and I/O handle lives here so that:
    //    • functions declare their dependencies explicitly via (SimState&),
    //    • multiple independent simulations can coexist, and
    //    • the struct can be serialised / checkpointed later.
    // =============================================================================

    #include "GridField.hpp"

    #include <algorithm>
    #include <string>
    #include <vector>
    #include <fstream>
    #include <filesystem>

    // _OPENMP is defined by the compiler only when built with -fopenmp
    // (standard, portable check) -- this header is shared between the
    // serial build (no -fopenmp, single-threaded) and the OpenMP build, so
    // the scratch-buffer sizing below sizes for 1 thread in the former and
    // omp_get_max_threads() in the latter.
    #ifdef _OPENMP
    #include <omp.h>
    #endif

    // ---------------------------------------------------------------------------
    //  A single (i,j,k) cell reference — used by red-black indexing
    //  (RedBlackIndexing.hpp) for the OpenMP/CUDA pressure-solve kernels.
    //  Defined here (not in RedBlackIndexing.hpp) so SimState can hold the
    //  built index lists without RedBlackIndexing.hpp and SimState.hpp
    //  including each other.
    // ---------------------------------------------------------------------------
    struct CellIndex { int i, j, k; };

    // ---------------------------------------------------------------------------
    //  Condition enumerations — replaces fragile string comparisons like "prd".
    // ---------------------------------------------------------------------------
    enum class GeometryType   { Axial, Curved };
    enum class GeometryShape  { AbruptExpansion, AbruptContraction, OpenCavity,
                                GradualExpansion, UnilateralExpansion,
                                GradualContraction, UnilateralContraction,
                                SharpCorner, RoundedCorner, Straight };
    enum class OutletBC       { ZeroFirstDeriv, ZeroSecondDeriv };
    enum class LateralBC      { Periodic, SolidWall };
    enum class InitialProfile { InletProfile, PotentialFlow };
    enum class FlowType       { SteadyMarching, RK4Transient };

    // ---------------------------------------------------------------------------
    //  Configuration  —  all user-settable parameters in one plain-old-data pod.
    //  These are filled from a config file or hard-coded defaults before INIT().
    // ---------------------------------------------------------------------------
    struct SimConfig {
        // Grid
        int numCellsX  = 120;   // II = 6*NN, NN=40
        int numCellsY  = 40;    // JJ = 2*NN
        int numCellsZ  = 20;    // KK = NN
        int baseUnit   = 20;    // NN

        // Physical domain dimensions (Cmp x Alt x Lrg in original)
        double domainLengthX = 6.0;   // Cmp
        double domainLengthY = 2.0;   // Alt
        double domainLengthZ = 1.0;   // Lrg

        // Cell sizes — derived from domain/grid; set automatically if left 0
        double cellSizeX = 0.0;   // dx = domainLengthX / numCellsX
        double cellSizeY = 0.0;   // dy = domainLengthY / numCellsY
        double cellSizeZ = 0.0;   // dz = domainLengthZ / numCellsZ

        // Physics
        double reynoldsNumber  = 100.0;
        double hyperViscousRe  = 20.0;
        int    hyperViscousStart = 0;

        // Time integration
        int    maxTimeSteps    = 10000;
        int    reportEveryN    = 1000;
        double convergenceTol  = 1e-6;

        // Boundary / geometry conditions  — matches NavSto_dynamic.cpp main()
        GeometryType   geometryType    = GeometryType::Curved;           // TipoGeometria = "Crv"
        GeometryShape  geometryShape   = GeometryShape::RoundedCorner;   // Geometria     = "Cam"
        OutletBC       outletCondition = OutletBC::ZeroFirstDeriv;        // CondSaida     = "1d0"
        LateralBC      lateralCondition = LateralBC::Periodic;            // CondLater     = "prd"
        InitialProfile initialProfile   = InitialProfile::PotentialFlow;  // PerfilInicial = "LCP"
        FlowType       flowType         = FlowType::SteadyMarching;       // itp = 0

        // Output
        std::filesystem::path outputDir  = "results";
        std::string           runName    = "cam_re9600";
        int                   numPressureIter = 5;

        // SOR relaxation factor for the pressure Poisson solve (1.0 = plain
        // Gauss-Seidel). 1.7 converges noticeably faster than 1.0 for this
        // grid/BC mix without the divergence risk of pushing closer to 2.0;
        // see docs/serial-optimization.md for how this was chosen/verified.
        double sorOmega = 1.7;
    };

    // ---------------------------------------------------------------------------
    //  SimState  —  the full mutable state of a running simulation.
    // ---------------------------------------------------------------------------
    struct SimState {

        // ── Configuration (set before allocateFields()) ───────────────────────────
        SimConfig cfg;

        // ── Derived / cached grid sizes ────────────────────────────────────────────
        GridSize g;
        int numCellsXm1 = 0, numCellsYm1 = 0, numCellsZm1 = 0;
        double cellSizeXsq = 0.0, cellSizeYsq = 0.0, cellSizeZsq = 0.0;

        // ── Geometry boundary index arrays ─────────────────────────────────────────
        std::vector<int> iLow, iHigh;   // leftmost/rightmost active i for row j
        std::vector<int> jLow, jHigh;   // lowest/highest active j for column i

        int degreeIndex1  = 0;
        int degreeIndex2  = 0;
        int degreeIndexY  = 0;
        int jLowInitial   = 0;
        int jHighFinal    = 0;
        int rampIndexX1   = 0;
        int rampIndexX2   = 0;
        int rampIndexY1   = 0;
        int rampIndexY2   = 0;
        int numActiveCells = 0;

        // ── Time integration state ─────────────────────────────────────────────────
        int    timeStep      = 0;
        int    midPlaneZ     = 0;
        int    iResidMax = 0, jResidMax = 0, kResidMax = 0;
        int    iDilMax   = 0, jDilMax   = 0, kDilMax   = 0;
        int    counter   = 0;
        bool   useHalfStep = false;   // true during RK sub-steps 1 & 2

        double timeStepSize    = 0.0;
        double simulationTime  = 0.0;

        double maxVelocityChange   = 0.0;
        double momentumResidMax    = 0.0;
        double momentumResidRMS    = 0.0;
        double dilatationMax       = 0.0;
        double intDivergence       = 0.0;
        double intAbsDivergence    = 0.0;
        double initialPressureGradX = 0.0;
        double initialPressureGradY = 0.0;
        double uMaxAtInlet  = 0.0;
        double vMaxAtInlet  = 0.0;
        double hyperViscousDecay = 0.0;

        // ── 3-D field arrays ───────────────────────────────────────────────────────
        GridField<> velX;            ///< u — x-velocity
        GridField<> velY;            ///< v — y-velocity
        GridField<> velZ;            ///< w — z-velocity
        GridField<> press;           ///< p — pressure
        GridField<> accelX;          ///< Au — UNIFAES acceleration on u
        GridField<> accelY;          ///< Av — UNIFAES acceleration on v
        GridField<> accelZ;          ///< Aw — UNIFAES acceleration on w
        GridField<> pressureSource;  ///< s  — RHS of pressure Poisson
        GridField<> scratchField;    ///< temporary (stream func., residuals…)

        // ── computeAccelerations() scratch buffers ──────────────────────────────────
        // Owned here (sized once in allocateFields()) instead of being
        // std::vector-allocated fresh on every computeAccelerations() call
        // (every timestep, and multiple times per step for RK4 substeps).
        // Safe to reuse without re-zeroing between calls: the active index
        // range each call touches is fixed by geometry (set once in
        // initSimulation() and never changed afterward), and every entry
        // in that range is written before it's read within the same call —
        // so whatever was left over from the previous call is always
        // overwritten before use. If that invariant ever changes (e.g. a
        // future adaptive/moving geometry), these need re-zeroing per call.
        // ppie/ppiw/qsie (the X-sweep's originally-1-D face-coefficient
        // buffers) were retired when the X-sweep moved to the 2-D ppieXK/
        // ppiwXK/qsieXK scratch below -- ppin/ppis/qsin (Y) and ppiu/ppid/
        // qsiu (Z) remain 1-D since those sweeps are unchanged. Ku/Kv/Kw
        // remain 1-D and shared across Y/Z (X now uses its own KuXK/KvXK/KwXK).
        //
        // Sized maxThreads * scratchLen (serial build: maxThreads=1, so
        // identical to before): the OpenMP build (src/openmp/Physics.cpp)
        // parallelizes each sweep's outer loop, and every thread needs its
        // own private slice of these buffers -- concurrent threads writing
        // the SAME shared buffer (as when maxThreads=1) would race. Each
        // thread's slice is threadId*scratchLen..+scratchLen (see
        // scratchLenPerThread below); the serial write-before-read-within-
        // one-call reasoning above still makes each thread's OWN slice safe
        // to reuse without re-zeroing.
        std::vector<double> ppin, ppis, ppiu, ppid;
        std::vector<double> qsin, qsiu, Ku, Kv, Kw;
        int scratchLenPerThread = 0;  // one thread's slice length for the buffers above

        // X-sweep-only 2-D scratch (logical i-index x k-index), flattened
        // per-thread as buf[threadId*xk2DLenPerThread + i_logical*scratchKLen + (k-1)].
        // See docs/serial-optimization-loop-order.md for why 2-D (k
        // innermost, matching GridField's storage); see the note above for
        // why per-thread (OpenMP parallelizes over j, the X-sweep's outer
        // loop, and every thread needs its own private slice).
        std::vector<double> ppieXK, ppiwXK, qsieXK, KuXK, KvXK, KwXK;
        int scratchKLen = 0;          // k-stride within one thread's (i,k) slice
        int xk2DLenPerThread = 0;     // one thread's slice length for the XK buffers above

        // ── Red-black pressure-solve indexing (OpenMP/CUDA only) ────────────────────
        // Built once (RedBlackIndexing.hpp::buildRedBlackIndices, lazily on
        // first use) and reused for the whole run -- geometry is fixed
        // after initSimulation(). Unused (empty) by the serial solver.
        std::vector<CellIndex> redCells, blackCells;
        bool redBlackBuilt = false;

        // ── Log / output ───────────────────────────────────────────────────────────
        std::ofstream logFile;

        // ── Memory management ──────────────────────────────────────────────────────
        /// Must be called once cfg.numCells* are set.
        void allocateFields() {
            // The physical domain is indices 0..numCells{X,Y,Z} (size
            // numCells+1), but the Gauss-Seidel pressure solve mirrors
            // Neumann boundary values one cell past the top of each axis
            // (e.g. s.press(numCellsX+1, j, k)) so the *next* sweep can read
            // a ghost value there. Allocate one extra ghost layer (+2, not
            // +1) so those writes land in valid, zero-initialized memory;
            // every loop elsewhere iterates the explicit 0..numCells range
            // and never sees the ghost layer.
            g.sI = cfg.numCellsX + 2;
            g.sJ = cfg.numCellsY + 2;
            g.sK = cfg.numCellsZ + 2;

            velX         .resize(g);
            velY         .resize(g);
            velZ         .resize(g);
            press        .resize(g);
            accelX       .resize(g);
            accelY       .resize(g);
            accelZ       .resize(g);
            pressureSource.resize(g);
            scratchField .resize(g);

            iLow .assign(g.sJ, 0);
            iHigh.assign(g.sJ, 0);
            jLow .assign(g.sI, 0);
            jHigh.assign(g.sI, 0);

            // computeAccelerations() scratch buffers — see field comments.
            // maxThreads: 1 for the serial build (no -fopenmp -> _OPENMP
            // undefined -> identical sizing/behavior to before per-thread
            // buffers existed); omp_get_max_threads() for the OpenMP build.
#ifdef _OPENMP
            const int maxThreads = omp_get_max_threads();
#else
            const int maxThreads = 1;
#endif
            // Cache-line padding (64 bytes = 8 doubles): round each
            // thread's slice length up to a multiple of 8 so adjacent
            // threads' slices don't share a cache line at their boundary
            // (mitigates false sharing on the frequently-written ends of
            // each slice; the vector's own base alignment isn't guaranteed
            // 64-byte, but this still separates thread boundaries cleanly).
            auto padTo8 = [](std::size_t n) { return (n + 7) & ~std::size_t{7}; };

            const int maxDim = std::max({cfg.numCellsX, cfg.numCellsY, cfg.numCellsZ});
            const std::size_t scratchLen = padTo8(static_cast<std::size_t>(maxDim) + 3);
            scratchLenPerThread = static_cast<int>(scratchLen);
            const std::size_t totalScratchLen = scratchLen * static_cast<std::size_t>(maxThreads);
            ppin.assign(totalScratchLen, 0.0); ppis.assign(totalScratchLen, 0.0);
            ppiu.assign(totalScratchLen, 0.0); ppid.assign(totalScratchLen, 0.0);
            qsin.assign(totalScratchLen, 0.0); qsiu.assign(totalScratchLen, 0.0);
            Ku  .assign(totalScratchLen, 0.0); Kv  .assign(totalScratchLen, 0.0); Kw  .assign(totalScratchLen, 0.0);

            // X-sweep 2-D scratch (see field comments): i-dimension sized
            // like the 1-D buffers above (maxDim+3, headroom for the +1
            // logical-index padding the VM() offset scheme needs); k
            // dimension sized numCellsZ+1 (covers k=1..numCellsZ, the
            // periodic-case upper bound for KKfim).
            const std::size_t iLen = static_cast<std::size_t>(maxDim) + 3;
            scratchKLen = cfg.numCellsZ + 1;
            const std::size_t xk2DLen = padTo8(iLen * static_cast<std::size_t>(scratchKLen));
            xk2DLenPerThread = static_cast<int>(xk2DLen);
            const std::size_t totalXK2DLen = xk2DLen * static_cast<std::size_t>(maxThreads);
            ppieXK.assign(totalXK2DLen, 0.0); ppiwXK.assign(totalXK2DLen, 0.0);
            qsieXK.assign(totalXK2DLen, 0.0);
            KuXK  .assign(totalXK2DLen, 0.0); KvXK  .assign(totalXK2DLen, 0.0); KwXK.assign(totalXK2DLen, 0.0);
        }
    };