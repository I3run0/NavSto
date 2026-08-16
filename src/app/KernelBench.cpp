// =============================================================================
//  KernelBench.cpp — time one per-step operator in isolation.
//
//  The inner loop of an optimization cycle. A whole simulation takes tens of
//  seconds and buries a kernel worth 5% of runtime under init, I/O and process
//  noise; this runs one kernel on a realistic state and reports ns/cell in
//  about a second. Whole-program benchmarking (scripts/benchmark.py) stays the
//  confirmation step.
//
//  State is built exactly as the solver builds it -- initSimulation() on a real
//  config -- so there is no fixture to keep in sync. JSON goes to stdout; the
//  Logger is quieted to ERR so it cannot interleave with it.
// =============================================================================

#include "SimState.hpp"
#include "Backend.hpp"
#include "Physics.hpp"
#include "KernelTimers.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string kernel = "all";
    int numCellsX = 96, numCellsY = 48, numCellsZ = 24;
    int iters = 20;
    int warmup = 3;
    int numPressureIter = 5;
    // Defaults to the production Reynolds number (experiments/configs/
    // production_can.cfg), NOT the 100.0 the smoke configs use. Re sets the
    // Peclet number, which selects the branch computeExponentialWeights takes
    // -- polynomial below 0.1, exp() up to 200, saturation above -- so
    // benchmarking at the wrong Re can measure a branch production never hits.
    double reynoldsNumber = 10000.0;
    // Geometry decides the i-span of each j-row, and several optimisations are
    // sensitive to span length (a peeled first iteration costs relatively more
    // on a short row). production_can.cfg runs RoundedCorner, not the
    // AbruptExpansion this used to hardcode.
    std::string shape = "RoundedCorner";
    int baseUnit = 40;
};

[[noreturn]] void usage(const char* argv0, int code) {
    std::cerr <<
        "usage: " << argv0 << " [options]\n"
        "  --kernel NAME   one of the seven operators, or 'all' (default: all)\n"
        "  --grid NXxNYxNZ grid size (default: 96x48x24)\n"
        "  --iters N       timed calls per kernel (default: 20)\n"
        "  --warmup N      untimed calls first (default: 3)\n"
        "  --pressure-iter N  numPressureIter (default: 5)\n"
        "  --re R          Reynolds number (default: 10000, the production value)\n"
        "  --shape NAME    AbruptExpansion|AbruptContraction|SharpCorner|\n"
        "                  RoundedCorner|Straight (default: RoundedCorner)\n"
        "  --base-unit N   geometry base unit NN (default: 40)\n";
    std::exit(code);
}

bool parseGrid(const std::string& v, Options& o) {
    // NXxNYxNZ
    const auto a = v.find('x');
    if (a == std::string::npos) return false;
    const auto b = v.find('x', a + 1);
    if (b == std::string::npos) return false;
    try {
        o.numCellsX = std::stoi(v.substr(0, a));
        o.numCellsY = std::stoi(v.substr(a + 1, b - a - 1));
        o.numCellsZ = std::stoi(v.substr(b + 1));
    } catch (...) { return false; }
    return o.numCellsX > 0 && o.numCellsY > 0 && o.numCellsZ > 0;
}

Options parseArgs(int argc, char* argv[]) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << what << "\n"; usage(argv[0], 2); }
            return argv[++i];
        };
        if      (a == "--kernel")        o.kernel = next("--kernel");
        else if (a == "--grid")        { if (!parseGrid(next("--grid"), o)) usage(argv[0], 2); }
        else if (a == "--iters")         o.iters = std::stoi(next("--iters"));
        else if (a == "--warmup")        o.warmup = std::stoi(next("--warmup"));
        else if (a == "--pressure-iter") o.numPressureIter = std::stoi(next("--pressure-iter"));
        else if (a == "--re")            o.reynoldsNumber = std::stod(next("--re"));
        else if (a == "--shape")         o.shape = next("--shape");
        else if (a == "--base-unit")     o.baseUnit = std::stoi(next("--base-unit"));
        else if (a == "-h" || a == "--help") usage(argv[0], 0);
        else { std::cerr << "unknown argument: " << a << "\n"; usage(argv[0], 2); }
    }
    if (o.iters < 1 || o.warmup < 0) usage(argv[0], 2);
    return o;
}

// ---------------------------------------------------------------------------
//  Which fields a kernel mutates, and so must be restored between timed calls.
//
//  Without this, solvePressurePoisson converges its own input away and
//  updateVelocities integrates the velocity field off into nonsense -- both
//  would still be "timeable" but would stop measuring the steady-state work the
//  solver actually does. The other five only write outputs nothing feeds back.
// ---------------------------------------------------------------------------
struct Snapshot {
    bool press = false, vel = false;
    std::vector<double> pressData, velXData, velYData, velZData;

    void capture(const SimState& s) {
        if (press) pressData = s.press.data();
        if (vel) { velXData = s.velX.data(); velYData = s.velY.data(); velZData = s.velZ.data(); }
    }
    void restore(SimState& s) const {
        if (press) s.press.data() = pressData;
        if (vel) { s.velX.data() = velXData; s.velY.data() = velYData; s.velZ.data() = velZData; }
    }
};

Snapshot snapshotFor(const std::string& kernel) {
    Snapshot snap;
    if (kernel == "solvePressurePoisson") snap.press = true;
    if (kernel == "updateVelocities")     snap.vel = true;
    return snap;
}

void callKernel(const std::string& k, SimState& s) {
    if      (k == "computeAccelerations")   computeAccelerations(s);
    else if (k == "buildPressureSource")    buildPressureSource(s);
    else if (k == "solvePressurePoisson")   solvePressurePoisson(s);
    else if (k == "updateVelocities")       updateVelocities(s);
    else if (k == "computeMomentumResidual")computeMomentumResidual(s);
    else if (k == "computeDivergence")      computeDivergence(s);
    else if (k == "adaptTimeStep")          adaptTimeStep(s);
}

struct Stats { double minMs, medianMs, meanMs, stddevMs; };

Stats summarize(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    const std::size_t n = ms.size();
    double sum = 0.0;
    for (double v : ms) sum += v;
    const double mean = sum / static_cast<double>(n);
    double sq = 0.0;
    for (double v : ms) sq += (v - mean) * (v - mean);
    const double stddev = (n > 1) ? std::sqrt(sq / static_cast<double>(n - 1)) : 0.0;
    const double median = (n % 2) ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
    return {ms.front(), median, mean, stddev};
}

/// Fresh state per kernel: a previous kernel's writes must not become this
/// one's input, or the measurement depends on benchmark ordering.
void buildState(SimState& s, const Options& o) {
    s.cfg.numCellsX = o.numCellsX;
    s.cfg.numCellsY = o.numCellsY;
    s.cfg.numCellsZ = o.numCellsZ;
    s.cfg.numPressureIter = o.numPressureIter;
    s.cfg.baseUnit = o.baseUnit;
    s.cfg.geometryShape =
          o.shape == "AbruptExpansion"   ? GeometryShape::AbruptExpansion
        : o.shape == "AbruptContraction" ? GeometryShape::AbruptContraction
        : o.shape == "SharpCorner"       ? GeometryShape::SharpCorner
        : o.shape == "Straight"          ? GeometryShape::Straight
                                         : GeometryShape::RoundedCorner;
    s.cfg.lateralCondition = LateralBC::SolidWall;
    s.cfg.outletCondition  = OutletBC::ZeroFirstDeriv;
    s.cfg.initialProfile   = InitialProfile::InletProfile;
    s.cfg.flowType         = FlowType::SteadyMarching;
    s.cfg.reynoldsNumber   = o.reynoldsNumber;
    s.cfg.hyperViscousStart = 0;
    s.cfg.cellSizeX = s.cfg.domainLengthX / s.cfg.numCellsX;
    s.cfg.cellSizeY = s.cfg.domainLengthY / s.cfg.numCellsY;
    s.cfg.cellSizeZ = s.cfg.domainLengthZ / s.cfg.numCellsZ;

    s.allocateFields();
    initSimulation(s);
    backendStartup(s);

    // Prime every kernel's inputs the way one solver step would: accelerations
    // exist, dt is set, and pressureSource is populated for the Poisson solve.
    computeAccelerations(s);
    adaptTimeStep(s);
    buildPressureSource(s);
}

}  // namespace

int main(int argc, char* argv[])
{
    const Options o = parseArgs(argc, argv);

    // Logger writes to stdout; keep it off so JSON is the only thing there.
    Logger::instance().setLevel(Logger::Level::ERR);

    std::vector<std::string> kernels;
    for (int i = 0; i < static_cast<int>(Kernel::COUNT); ++i)
        kernels.emplace_back(kernelName(static_cast<Kernel>(i)));

    if (o.kernel != "all") {
        if (std::find(kernels.begin(), kernels.end(), o.kernel) == kernels.end()) {
            std::cerr << "unknown kernel: " << o.kernel << "\n";
            return 2;
        }
        kernels = {o.kernel};
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "{\n  \"backend\": \"" << backendName() << "\",\n";
    std::cout << "  \"grid\": \"" << o.numCellsX << "x" << o.numCellsY << "x" << o.numCellsZ << "\",\n";
    std::cout << "  \"iters\": " << o.iters << ",\n";
    std::cout << "  \"warmup\": " << o.warmup << ",\n";
    std::cout << "  \"numPressureIter\": " << o.numPressureIter << ",\n";
    std::cout << "  \"reynoldsNumber\": " << o.reynoldsNumber << ",\n";
    std::cout << "  \"shape\": \"" << o.shape << "\",\n";

    long long activeCells = 0;
    std::vector<std::string> records;

    for (const auto& k : kernels) {
        SimState s;
        buildState(s, o);
        activeCells = s.numActiveCells;

        Snapshot snap = snapshotFor(k);
        snap.capture(s);

        for (int i = 0; i < o.warmup; ++i) { snap.restore(s); callKernel(k, s); }

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(o.iters));
        for (int i = 0; i < o.iters; ++i) {
            snap.restore(s);                       // untimed: outside the window
            const auto t0 = Clock::now();
            callKernel(k, s);
            const auto t1 = Clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        const Stats st = summarize(samples);
        const double nsPerCell = activeCells > 0
            ? st.minMs * 1e6 / static_cast<double>(activeCells) : 0.0;

        std::string rec = "    {\"kernel\": \"" + k + "\"";
        auto num = [](double v) {
            std::ostringstream os; os << std::fixed << std::setprecision(9) << v; return os.str();
        };
        rec += ", \"min_ms\": "    + num(st.minMs);
        rec += ", \"median_ms\": " + num(st.medianMs);
        rec += ", \"mean_ms\": "   + num(st.meanMs);
        rec += ", \"stddev_ms\": " + num(st.stddevMs);
        rec += ", \"rel_spread\": " + num(st.meanMs > 0 ? st.stddevMs / st.meanMs : 0.0);
        rec += ", \"ns_per_active_cell\": " + num(nsPerCell);
        rec += ", \"restored_between_iters\": " + std::string((snap.press || snap.vel) ? "true" : "false");
        rec += "}";
        records.push_back(rec);

        backendShutdown(s);
    }

    std::cout << "  \"active_cells\": " << activeCells << ",\n";
    std::cout << "  \"kernels\": [\n";
    for (std::size_t i = 0; i < records.size(); ++i)
        std::cout << records[i] << (i + 1 < records.size() ? ",\n" : "\n");
    std::cout << "  ]\n}\n";

    return 0;
}
