#!/usr/bin/env python3
"""autoresearch.py — a speedrun harness for NavSolver.

One number to beat, an append-only chain of records, and a referee that cannot
be talked around. The agent writes the code; this decides whether it counts.

    python3 scripts/autoresearch.py status        # the record, and the target it is against
    python3 scripts/autoresearch.py next          # ranked idea queue
    python3 scripts/autoresearch.py attempt -m "what I changed"
    python3 scripts/autoresearch.py leaderboard   # the chain
    python3 scripts/autoresearch.py log  -m "..." # record a negative result

TWO TRACKS, because this project has two kinds of win and one metric cannot
judge both.

  perf   The answer must not move. Bit-identical output across the whole
         configuration matrix, then fastest wall clock wins. This is where
         loop restructuring, vectorisation and scheduling live.

  method The answer is allowed to move, and must move in a defensible
         direction: same or less time for strictly better incompressibility.
         Judged on the (time, IntAbsDiv, DilMax) Pareto front, because the
         pressure solve today never checks convergence -- it runs a fixed
         sweep count -- so "faster" and "more converged" are separate axes and
         a change that trades one for the other is not progress.

WHY THE REFEREE LOOKS THE WAY IT DOES. Three things this machine does that
will produce wrong verdicts if you do not control for them:

  * Code placement moves untouched kernels by several percent -- updateVelocities
    has been measured at 0.695-1.359x between functionally identical builds. So
    every attempt disassembles both binaries and treats any kernel whose machine
    code is byte-identical as a CONTROL: its movement is placement by definition
    and never counts for or against.

  * Cross-session drift exceeds within-session noise. Stored baselines have put
    byte-identical kernels at 0.94x and 1.30x forty minutes apart. So every
    comparison is interleaved against a freshly built record binary, in one
    session, never against a saved number.

  * Multi-threaded timing is not a signal here at all. OpenMP attempts run at
    one thread; threading speedup is a whole-program question, not a kernel one.
"""

import argparse, datetime as dt, json, os, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import REPO_ROOT, BENCH_ENV, cpu_model  # noqa: E402

DIR       = REPO_ROOT / "experiments" / "research"
RECORDS   = DIR / "records.json"
LEDGER    = DIR / "ledger.md"
BACKLOG   = DIR / "backlog.md"

# ── The target. Pinned here so it cannot drift between attempts. ────────────
# Production physics (RoundedCorner, Re=10000, RK4, periodic z) at a grid small
# enough to run in a minute and large enough that fields exceed L3 -- the regime
# production actually runs in, not the cache-resident one kbench measures.
TARGET = dict(grid=(180, 60, 30), steps=400, re=10000.0, pressure_iter=10)
TARGET_CFG = """numCellsX = 180
numCellsY = 60
numCellsZ = 30
domainLengthX = 9.0
domainLengthY = 3.0
domainLengthZ = 1.5
reynoldsNumber = 10000.0
hyperViscousRe = 500.0
hyperViscousStart = 150
flowType = RK4Transient
maxTimeSteps = 400
convergenceTol = 0
numPressureIter = 10
reportEveryN = 400
geometryShape = RoundedCorner
geometryType = Curved
lateralBC = Periodic
outletBC = ZeroFirstDeriv
initialProfile = PotentialFlow
"""

KERNELS = ["computeAccelerations", "buildPressureSource", "solvePressurePoisson",
           "updateVelocities", "computeMomentumResidual", "computeDivergence",
           "adaptTimeStep"]

# ── Accuracy matrix: every branch that has ever hidden a bug. ───────────────
_BASE = """numCellsX = 64
numCellsY = 32
numCellsZ = 16
reynoldsNumber = 400.0
hyperViscousStart = 0
geometryShape = AbruptExpansion
lateralBC = SolidWall
outletBC = ZeroFirstDeriv
initialProfile = InletProfile
flowType = RK4Transient
maxTimeSteps = 60
convergenceTol = 0
numPressureIter = 5
reportEveryN = 10
"""

def _cfg(**over):
    out, seen = [], set()
    for line in _BASE.strip().split("\n"):
        k = line.split("=")[0].strip()
        seen.add(k)
        out.append(f"{k} = {over[k]}" if k in over else line)
    out += [f"{k} = {v}" for k, v in over.items() if k not in seen]
    return "\n".join(out) + "\n"

CASES = {
    "solid":      _cfg(),
    "perRK4":     _cfg(lateralBC="Periodic"),
    "perSteady":  _cfg(lateralBC="Periodic", flowType="SteadyMarching"),
    "perRound":   _cfg(lateralBC="Periodic", geometryShape="RoundedCorner", baseUnit=12),
    "perHyper":   _cfg(lateralBC="Periodic", hyperViscousStart=16),
    "nz1":        _cfg(lateralBC="Periodic", numCellsZ=1),   # degenerate z: the
    "nz2":        _cfg(lateralBC="Periodic", numCellsZ=2),   # periodic-wrap peels
    "nz3":        _cfg(lateralBC="Periodic", numCellsZ=3),   # special-case these
    "mgSolid":    _cfg(pressureSolver="Multigrid"),
    "mgPerRound": _cfg(lateralBC="Periodic", geometryShape="RoundedCorner",
                       baseUnit=12, pressureSolver="Multigrid"),
}


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def load_records():
    return json.loads(RECORDS.read_text()) if RECORDS.exists() else []


def save_records(recs):
    DIR.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text(json.dumps(recs, indent=2) + "\n")


def git_short(ref="HEAD"):
    return sh(["git", "rev-parse", "--short", ref], cwd=REPO_ROOT).stdout.strip()


def build_ref(ref, dest, targets):
    wt = dest / "src"
    sh(["git", "worktree", "add", "-f", "--detach", str(wt), ref], cwd=REPO_ROOT)
    bld = wt / "build"
    sh(["cmake", "-S", str(wt), "-B", str(bld), "-DCMAKE_BUILD_TYPE=Release"])
    r = sh(["cmake", "--build", str(bld), "-j", "--target", *targets])
    if r.returncode:
        raise SystemExit("record-holder build failed:\n" + (r.stdout + r.stderr)[-2000:])
    return bld


def run_target(binary, workdir, threads=None):
    """One timed run of the target workload. Returns (seconds, quality dict)."""
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = workdir / "run.cfg"
    cfg.write_text(f"outputDir = {workdir}\nrunName = t\n" + TARGET_CFG)
    env = {**os.environ, **BENCH_ENV}
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    t0 = dt.datetime.now()
    r = subprocess.run([str(binary), str(cfg)], capture_output=True, text=True, env=env)
    secs = (dt.datetime.now() - t0).total_seconds()
    if r.returncode:
        raise SystemExit(f"target run failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    rows = [l.split(",") for l in (workdir / "t_convergence.csv").read_text().split("\n")[1:] if l]
    tail = [[float(x) for x in row] for row in rows][-100:]
    return secs, {"IntAbsDiv": statistics.mean(r[6] for r in tail),
                  "DilMax":    statistics.mean(r[4] for r in tail)}


def paired_target(base_bin, cand_bin, tmp, repeats, threads=None):
    """Interleaved, alternating which goes first. Median of paired ratios and a
    win count -- read both: 7/7 at 1.02x is a result, 4/7 at 1.05x is not."""
    ratios, base_t, cand_t, qual = [], [], [], {}
    for i in range(repeats):
        order = [("base", base_bin), ("cand", cand_bin)]
        if i % 2:
            order.reverse()
        got = {}
        for tag, b in order:
            got[tag] = run_target(b, tmp / f"tgt_{tag}_{i}", threads)
        base_t.append(got["base"][0]); cand_t.append(got["cand"][0])
        ratios.append(got["base"][0] / got["cand"][0])
        qual = {"base": got["base"][1], "cand": got["cand"][1]}
    return {"ratios": ratios, "median": statistics.median(ratios),
            "wins": sum(1 for r in ratios if r > 1.0), "n": len(ratios),
            "base_best": min(base_t), "cand_best": min(cand_t), "quality": qual}


def symbol_map(binary):
    """Every function symbol -> its instruction stream, addresses normalised so
    two copies of identical code compare equal wherever the linker put them.

    Whole-binary, not a fixed list of kernel names: the hot code often lives in
    a static helper (gaussSeidelSweeps behind solvePressurePoisson, the row
    kernels behind computeAccelerations), and inspecting only the named entry
    points reports a rewritten smoother as an unchanged control."""
    r = sh(["objdump", "-d", "--no-show-raw-insn", "-C", str(binary)])
    syms, cur = {}, None
    for line in r.stdout.split("\n"):
        m = re.match(r"^[0-9a-f]+ <(.+)>:", line)
        if m:
            cur = m.group(1)
            syms[cur] = []
            continue
        if cur and line.strip():
            s = re.sub(r"^\s*[0-9a-f]+:\s*", "", line)
            s = re.sub(r"0x[0-9a-f]+", "H", s)
            syms[cur].append(re.sub(r"\b[0-9a-f]{4,}\b", "A", s))
    return syms


def classify(base_bin, cand_bin):
    """Split symbols into what this edit actually touched and what it did not.

    A kernel counts as touched if its own code changed OR if anything it calls,
    transitively, changed -- solvePressurePoisson's body is a two-line dispatch
    and the smoother it calls is a separate static symbol, so a rewritten
    smoother would otherwise be filed as an unchanged control and its win
    thrown away. Only what survives that closure is a genuine control."""
    a, b = symbol_map(base_bin), symbol_map(cand_bin)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}

    calls = {}                       # symbol -> symbols it calls
    for sym, insns in a.items():
        tgt = set()
        for ins in insns:
            m = re.search(r"call.*<([^>+]+)", ins)
            if m:
                tgt.add(m.group(1))
        calls[sym] = tgt

    tainted, frontier = set(changed), list(changed)
    while frontier:                  # propagate callee -> caller to a fixpoint
        cur = frontier.pop()
        for sym, tgts in calls.items():
            if sym not in tainted and cur in tgts:
                tainted.add(sym)
                frontier.append(sym)

    kernels_hit = [k for k in KERNELS if any(k in s for s in tainted)]
    others = sorted(s for s in changed if not any(k in s for k in KERNELS))
    controls = [k for k in KERNELS if k not in kernels_hit]
    return kernels_hit, others, controls


def accuracy(base_bin, cand_bin, tmp, threads=None, skip_csv=False):
    res = {}
    for name, cfg in CASES.items():
        paths, failed = {}, False
        for tag, binary in (("base", base_bin), ("cand", cand_bin)):
            d = tmp / f"acc_{name}_{tag}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            (d / "run.cfg").write_text(f"outputDir = {d}\nrunName = t\n" + cfg)
            env = {**os.environ, **BENCH_ENV}
            if threads:
                env["OMP_NUM_THREADS"] = str(threads)
            if subprocess.run([str(binary), str(d / "run.cfg")],
                              capture_output=True, env=env).returncode:
                res[name] = {"ok": False, "differing": ["run failed"], "files": 0}
                failed = True
                break
            paths[tag] = d
        if failed:
            continue
        diffs, n = [], 0
        for f in sorted(paths["base"].glob("t*")):
            if f.suffix == ".cfg" or (skip_csv and f.name.endswith("convergence.csv")):
                continue
            n += 1
            if f.read_bytes() != (paths["cand"] / f.name).read_bytes():
                diffs.append(f.name)
        res[name] = {"ok": not diffs, "files": n, "differing": diffs}
    return res


# ── commands ───────────────────────────────────────────────────────────────

def cmd_status(a):
    recs = load_records()
    nx, ny, nz = TARGET["grid"]
    print(f"target   : {nx}x{ny}x{nz}, {TARGET['steps']} RK4 steps, Re={TARGET['re']:.0f}, "
          f"numPressureIter={TARGET['pressure_iter']}")
    print(f"machine  : {cpu_model()}")
    if not recs:
        print("record   : none yet — run `attempt` to set the first one")
        return
    r = recs[-1]
    print(f"record   : {r['seconds']:.2f} s  ({r['track']})  {r['commit']}  {r['date']}")
    print(f"           {r['message']}")
    print(f"quality  : IntAbsDiv={r['quality']['IntAbsDiv']:.4f}  DilMax={r['quality']['DilMax']:.2f}")
    print(f"chain    : {len(recs)} record(s); first was {recs[0]['seconds']:.2f} s "
          f"({recs[0]['seconds']/r['seconds']:.3f}x total)")


def cmd_leaderboard(a):
    recs = load_records()
    if not recs:
        print("(no records yet)")
        return
    print(f"{'#':>3}  {'seconds':>8} {'vs prev':>8} {'track':>7}  {'commit':<9} message")
    prev = None
    for i, r in enumerate(recs, 1):
        gain = f"{prev/r['seconds']:.3f}x" if prev else "—"
        print(f"{i:>3}  {r['seconds']:8.2f} {gain:>8} {r['track']:>7}  {r['commit']:<9} {r['message']}")
        prev = r["seconds"]
    print(f"\ntotal: {recs[0]['seconds']/recs[-1]['seconds']:.3f}x over {len(recs)} records")


def cmd_next(a):
    print(BACKLOG.read_text() if BACKLOG.exists() else "(no backlog)")


def cmd_log(a):
    DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    hdr = "" if LEDGER.exists() else ("# Research ledger\n\nAppend-only. Negative results belong "
                                      "here too — a rejected idea nobody wrote down gets retried.\n")
    LEDGER.write_text((LEDGER.read_text() if LEDGER.exists() else hdr)
                      + f"\n## {stamp}Z — note ({git_short()})\n{a.message}\n")
    print(f"logged to {LEDGER.relative_to(REPO_ROOT)}")


def cmd_attempt(a):
    DIR.mkdir(parents=True, exist_ok=True)
    recs = load_records()
    ref = a.against or (recs[-1]["commit"] if recs else "HEAD")
    omp = a.backend == "openmp"
    solver = "navsolver_omp" if omp else "navsolver"

    print(f"[build]    candidate = working tree; record holder = {ref}")
    if sh(["cmake", "--build", str(REPO_ROOT / "build"), "-j", "--target", solver],
          cwd=REPO_ROOT).returncode:
        raise SystemExit("candidate build failed")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base_bld = build_ref(ref, tmp, [solver])
        base_bin, cand_bin = base_bld / solver, REPO_ROOT / "build" / solver
        try:
            print(f"[accuracy] {len(CASES)} configurations, byte-comparing every output")
            acc = accuracy(base_bin, cand_bin, tmp, a.threads,
                           skip_csv=omp and (a.threads or 0) != 1)
            bit_identical = all(v["ok"] for v in acc.values())
            for n, v in acc.items():
                print(f"    {'ok  ' if v['ok'] else 'DIFF'} {n:<11} {v['files']} files"
                      + ("" if v["ok"] else f"  -> {v['differing']}"))

            print("[control]  diffing every function symbol in both binaries")
            changed, other_syms, controls = classify(base_bin, cand_bin)
            print(f"    kernels changed : {', '.join(changed) or '(none)'}")
            if other_syms:
                shown = ", ".join(other_syms[:6]) + (f" (+{len(other_syms)-6} more)"
                                                     if len(other_syms) > 6 else "")
                print(f"    other symbols   : {shown}")
            print(f"    controls        : {', '.join(controls) or '(none)'}   "
                  f"(identical code; their movement is placement and never counts)")

            print(f"[timing]   {a.repeats} interleaved pairs on the target workload")
            p = paired_target(base_bin, cand_bin, tmp, a.repeats, a.threads)
            for i, r in enumerate(p["ratios"], 1):
                print(f"    pair {i}: {r:.3f}x")
            print(f"    median {p['median']:.3f}x  ({p['wins']}/{p['n']} pairs favour candidate)"
                  f"   spread {min(p['ratios']):.3f}-{max(p['ratios']):.3f}x")
            print(f"    record holder best {p['base_best']:.2f} s, candidate best {p['cand_best']:.2f} s")
            qb, qc = p["quality"]["base"], p["quality"]["cand"]
            print(f"    quality  base IntAbsDiv={qb['IntAbsDiv']:.4f} DilMax={qb['DilMax']:.2f}"
                  f" | cand IntAbsDiv={qc['IntAbsDiv']:.4f} DilMax={qc['DilMax']:.2f}")
        finally:
            sh(["git", "worktree", "remove", "--force", str(tmp / "src")], cwd=REPO_ROOT)
            sh(["git", "worktree", "prune"], cwd=REPO_ROOT)

    # ── adjudicate ─────────────────────────────────────────────────────────
    faster = p["median"] > 1.0 and p["wins"] >= (2 * p["n"]) // 3
    better = (qc["IntAbsDiv"] <= qb["IntAbsDiv"] and qc["DilMax"] <= qb["DilMax"]
              and (qc["IntAbsDiv"] < qb["IntAbsDiv"] or qc["DilMax"] < qb["DilMax"]))
    slower = p["median"] < 1.0 and p["wins"] <= p["n"] // 3

    if a.track == "perf":
        if not bit_identical:
            verdict, why = "REJECTED", "answer moved; a perf record must be bit-identical"
        elif not faster:
            verdict, why = "REJECTED", f"not resolvably faster ({p['median']:.3f}x, {p['wins']}/{p['n']})"
        else:
            verdict, why = "RECORD", f"{p['median']:.3f}x ({p['wins']}/{p['n']} pairs), answer unchanged"
    else:  # method
        if bit_identical:
            verdict, why = "REJECTED", "answer did not move; submit this as a perf attempt"
        elif slower and not better:
            verdict, why = "REJECTED", f"slower ({p['median']:.3f}x) and not more converged"
        elif not better and not faster:
            verdict, why = "REJECTED", "neither more converged nor faster"
        else:
            verdict, why = "RECORD", (f"{p['median']:.3f}x, IntAbsDiv "
                                      f"{qb['IntAbsDiv']:.4f}->{qc['IntAbsDiv']:.4f}, "
                                      f"DilMax {qb['DilMax']:.2f}->{qc['DilMax']:.2f}")

    print(f"\n=== {verdict} — {why}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    hdr = "" if LEDGER.exists() else ("# Research ledger\n\nAppend-only. Negative results belong "
                                      "here too — a rejected idea nobody wrote down gets retried.\n")
    LEDGER.write_text((LEDGER.read_text() if LEDGER.exists() else hdr) + "\n".join([
        f"\n## {stamp}Z — {verdict} ({a.track})",
        f"- **Attempt:** {a.message}",
        f"- **Against:** `{ref}`   **Backend:** {a.backend}"
        + (f" @ {a.threads}T" if a.threads else ""),
        f"- **Bit-identical:** {bit_identical}"
        + ("" if bit_identical else f"  ({[k for k,v in acc.items() if not v['ok']]})"),
        f"- **Code changed in:** {', '.join(changed) or '(none)'}",
        f"- **Timing:** {p['median']:.3f}x, {p['wins']}/{p['n']} pairs; "
        f"{p['base_best']:.2f} s -> {p['cand_best']:.2f} s",
        f"- **Quality:** IntAbsDiv {qb['IntAbsDiv']:.4f} -> {qc['IntAbsDiv']:.4f}, "
        f"DilMax {qb['DilMax']:.2f} -> {qc['DilMax']:.2f}",
        f"- **Verdict:** {why}", ""]))

    if verdict == "RECORD" and not a.dry_run:
        recs.append({"date": stamp, "commit": git_short(), "track": a.track,
                     "seconds": p["cand_best"], "message": a.message,
                     "quality": qc, "ratio_vs_prev": p["median"],
                     "machine": cpu_model()})
        save_records(recs)
        print(f"[record]   chain now {len(recs)} deep; commit the change, then re-run "
              f"`status` to confirm the commit hash")
    return 0 if verdict == "RECORD" else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="cmd", required=True)
    s.add_parser("status").set_defaults(fn=cmd_status)
    s.add_parser("leaderboard").set_defaults(fn=cmd_leaderboard)
    s.add_parser("next").set_defaults(fn=cmd_next)

    lg = s.add_parser("log"); lg.add_argument("-m", "--message", required=True)
    lg.set_defaults(fn=cmd_log)

    at = s.add_parser("attempt")
    at.add_argument("-m", "--message", required=True, help="what was changed, in one line")
    at.add_argument("--track", default="perf", choices=("perf", "method"))
    at.add_argument("--against", default=None, help="git ref to beat (default: current record)")
    at.add_argument("--backend", default="serial", choices=("serial", "openmp"))
    at.add_argument("--threads", type=int, default=None,
                    help="OpenMP threads; use 1 — multi-thread timing is not a signal here")
    at.add_argument("--repeats", type=int, default=7)
    at.add_argument("--dry-run", action="store_true", help="adjudicate but do not claim the record")
    at.set_defaults(fn=cmd_attempt)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
