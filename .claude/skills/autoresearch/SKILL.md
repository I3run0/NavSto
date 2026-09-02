---
name: autoresearch
description: Run one performance-optimization hypothesis through the full experimental loop on NavSolver — attribute where time goes, state a mechanism, measure it in a way this machine cannot fake, adjudicate against the strict Pareto gate, and record the outcome including failures. Use when optimizing any backend, when a benchmark result looks surprising, when deciding what to optimize next, or when a measurement needs to be trusted before it is acted on.
---

# The autoresearch performance loop

This is an **experiment protocol**, not a benchmark runner. The output of one
iteration is a decided hypothesis and a ledger entry, not a number.

The premise is that on this hardware **the measurement is the hard part**. Every
rule below exists because a specific plausible-looking measurement here was
wrong, and the literature says why.

## 0. The frame

Treat each iteration as a preregistered experiment.

- **One hypothesis per iteration.** State the *mechanism* you expect to pay off
  *before* measuring — "the branch chain blocks vectorisation of ~40 FLOPs" —
  not "this might be faster." Writing the mechanism down first is the defence
  against HARKing (hypothesizing after results are known): a post-hoc story
  fits any outcome, so a hypothesis that was never falsifiable teaches nothing.
- **Predict the size.** If you expect 5% and get 40%, something else changed —
  usually the build, the config, or the machine. Large agreement with a
  prediction is evidence; large disagreement is a bug until proven otherwise.
- **Failures are the product.** A rejected hypothesis with a number attached is
  worth more than an accepted one with a vague one, because it prunes the
  search space permanently. Record it (§6).

## 1. What this machine does to measurements

### Code layout bias

Two builds that differ only in an appended, never-called function measured
`updateVelocities` **0.695x to 1.359x** apart. `computeAccelerations` stays
inside 0.992–1.013x. The effect is real, per-kernel, and can exceed the effect
you are trying to measure.

This is the canonical measurement-bias result: Mytkowicz, Diwan, Hauswirth &
Sweeney, *"Producing Wrong Data Without Doing Anything Obviously Wrong!"*
(ASPLOS 2009) showed link order and environment size alone can produce
speedups larger than the optimizations under study. Curtsinger & Berger's
**STABILIZER** (ASPLOS 2013) answers it by randomizing code layout so layout
becomes noise you can average over rather than bias you cannot see.

**We use a cheaper equivalent.** Rather than randomize layout, identify code
that *cannot* have changed: diff every function symbol between the two
binaries and propagate through the call graph to a fixpoint. A symbol whose
instruction stream is byte-identical is a **control** — its movement is
placement by definition. This is strictly stronger than a layout band, and it
costs one `objdump`.

> The call-graph closure is not optional. `solvePressurePoisson`'s body is a
> two-line dispatch; its smoother is a separate static symbol. Comparing only
> the named kernel filed a fully rewritten smoother as an unchanged control.

### Cross-session drift

This machine drifts more between sessions than its within-session noise floor.
Stored-baseline comparisons put **byte-identical** kernels at 0.94x and then
1.30x forty minutes apart, and a whole-program tier flagged a *different*
benchmark size as regressed on two consecutive runs.

**Rule: never adjudicate against a stored number.** Build both arms fresh in
one session and interleave them, alternating which runs first, so drift lands
on both equally. Report the **median of paired ratios and the win count** —
7/7 at 1.02x is a result; 4/7 at 1.05x is not. This is the paired/blocked
design from classical experiment theory; for the performance-specific version
see Georges, Buytaert & Eeckhout, *"Statistically Rigorous Java Performance
Evaluation"* (OOPSLA 2007), which is the standard argument for repeated runs
and intervals over single-number comparisons.

### Multiple comparisons

Seven kernels × three grids = 21 comparisons per attempt. Some will look
significant by chance. The control analysis is what keeps this honest: it
collapses the family to *only* the kernels the edit could have reached,
usually one or two. Never harvest a win from a kernel the diff did not touch.

### Multi-threaded timing

Byte-identical binaries measured **1.85x apart at 6 threads, 7/7 pairs**, while
tying to within 1% at one thread. Whole-program six-thread runs gave 0.948x
(0/7) in one window and 0.974x (5/11) in another, and bisecting the interval
put *both halves* above 1.0x — arithmetically impossible if the whole is below.

**Rule: measure OpenMP *code* changes at `--threads 1`.** Threading speedup is
a separate, whole-program question. If you cannot resolve it, say so rather
than reporting a number.

## 2. Characterize before optimizing — this is phase 0, not optional

**Never pick a hypothesis before running `characterize`.** Every large win in
this project came from characterization overturning a belief, not from having a
good idea: the roofline implied the transcendental and the answer was
divisions; the documentation said memory-bound and a working-set sweep said no.
Optimizing without a current characterization is guessing with extra steps.

```bash
python3 scripts/autoresearch.py characterize          # all probes
python3 scripts/autoresearch.py characterize --probe divisions gs-recurrence
```

It writes `experiments/research/profile.md` with four things:

1. **The workload.** Grid, active vs ghosted cells, working-set size against
   this machine's L2/L3, and therefore the **regime**. This matters more than
   it looks: `kbench`'s grids are cache-resident and the production grid is
   not, so a win measured in one regime need not transfer. Check the regime
   line before believing a result generalises.
2. **Attribution.** Per-kernel share of step time.
3. **A working-set sweep.** ns per active cell as the set grows past the caches.
   A bandwidth-bound kernel falls off a cliff there; a latency- or
   throughput-bound one barely moves.
4. **Causal probes.** What speeding each construct up would actually be worth.

### Probe validity — the part that bites

A probe is a same-shape stand-in built from the same commit, and it is only
meaningful if it is **on the hot path**. Matching the source text does not
establish that. A probe pointed at `computeExponentialWeights` — a function the
division work had moved off the host hot path, but which still existed and
still matched — compiled cleanly, reported 15.1% at Re=10000, and was measuring
pure noise. The physically impossible part gave it away: at Re=10000 `DPe` is
around 625, the `|DPe| > 200` branch returns without calling `exp` at all, so
the true share is ~0.

So every probe is checked for **inertness**: run the real and stand-in binaries
on one configuration and compare output. Deliberately wrong arithmetic that
changes *no output* is not on the hot path, and the probe reports INERT rather
than a number. Probes also report STALE when their source text no longer
matches. Treat either as a bug in the probe registry to fix before ranking
anything against it.

Two further cautions:

- **Report the spread.** The same probe measured 8.7% and 0.2% on consecutive
  runs. A share quoted without a range invites ranking a backlog against noise.
- **Shares do not sum.** Removing one division lets the others' latency
  overlap, so each measured alone looks larger than its marginal contribution.
  A probe share is an upper bound on what removing that construct buys — and
  removing it is usually not legal. The value is knowing what to restructure
  *around*.

## 3. Attribution instruments — find the real constraint before optimizing

Do not guess. Four instruments, cheapest first.

### Roofline

Williams, Waterman & Patterson, *"Roofline: An Insightful Visual Performance
Model for Multicore Architectures"* (CACM 2009). Bounds achievable performance
by arithmetic intensity against measured ceilings.

`scripts/roofline.py` plus the microbenchmarks in `scripts/roofline/`
(`peak_flops`, `peak_bandwidth`, `peak_exp_throughput`,
`peak_bandwidth_cuda`). **Measure the ceiling, never quote a spec sheet** — and
tune the microbenchmark itself: a first, under-occupied GPU bandwidth kernel
read 29 GB/s where a swept configuration reads 90.4 GB/s, which would have
supported the exactly wrong conclusion.

Roofline gives a *bound*, not an attribution. Read a derived ceiling
(`exp_ceiling_gflops_1t`) as a lower bound on achievable, never as "this is
where the time goes" — that specific misreading sent one iteration after a
transcendental that turned out to be ≤7% of the kernel.

### Working-set sweep — is it bandwidth at all?

Sweep grid size and watch ns/active-cell. `computeAccelerations` moved only
**17% across a 145x growth** in working set, well past a 12 MB L3. A
bandwidth-bound kernel falls off a cliff there; this one does not. Two lines
of shell, and it falsified the standing "memory/cache-access-pattern" theory.

Note the regime changes with size: `kbench` grids are cache-resident, the
production grid is not. Verify a win transfers.

### Causal-style stand-in builds — the strongest attribution available here

Build a variant from the *same commit* with one construct replaced by a
same-shape stand-in of the same operation count, deliberately wrong results,
never committed. Time it interleaved against the real binary. The delta is
that construct's share.

This is a manual form of **causal profiling** — Curtsinger & Berger, *"Coz:
Finding Code that Counts with Causal Profiling"* (SOSP 2015). Coz's insight is
that conventional profilers report where time is *spent*, which is not the same
as where speeding code up would *help*; it answers the counterfactual by
virtually speeding a region up. The stand-in build answers the same
counterfactual directly.

Measured this way, on `computeAccelerations`:

| construct removed | share |
|---|---|
| `exp()` | 1.6–3.0% (Re=10000), 6.7–7.1% (Re=100) |
| all divisions | ~29% |
| scalar operand pass (branches + exp) | 21–26% |

The `exp` row overturned the roofline reading and redirected the work that
produced the session's largest win. **Run at every Reynolds number the project
uses** — `kbench` defaults to Re=10000 and the reward workload uses Re=100, and
they take different branches.

### Little's Law — latency-bound or throughput-bound?

Concurrency = latency × throughput. When a dependent chain dominates, more
arithmetic units do not help; more *independent work in flight* does. Break the
recurrence in a diagnostic build (read forward instead of backward, same
operation count) and time it: the Gauss-Seidel k-recurrence measured **69–75%**
of `solvePressurePoisson`, which justified the diagonal-blocking rewrite that
cut it to 24–31%.

Ousterhout's *"Always Measure One Level Deeper"* (CACM 2018) is the general
form of all four: the top-line number tells you nothing about mechanism, and
mechanism is what you optimize.

## 4. The loop

```
characterize → next → pick one hypothesis → state the mechanism → implement →
  attempt ─┬─ RECORD  → commit alone, re-point the record, update the backlog
           └─ REJECT  → revert the tree, log the number and why, close or refine
```

```bash
python3 scripts/autoresearch.py characterize   # phase 0 — where the time is and why
python3 scripts/autoresearch.py status        # the record and the target it stands against
python3 scripts/autoresearch.py next          # ranked idea queue
python3 scripts/autoresearch.py attempt -m "one line: what changed"
python3 scripts/autoresearch.py leaderboard   # the chain of records
python3 scripts/autoresearch.py log -m "..."  # a negative result worth keeping
```

`attempt` runs the fixed protocol with no discretion:

1. Build **both** arms fresh in this session.
2. **Accuracy matrix** — 10 configurations covering both lateral BCs, both flow
   types, both geometries, the hyper-viscous sponge, degenerate `numCellsZ ∈
   {1,2,3}`, and the multigrid path. Byte-compare every output file.
3. **Attribution** — symbol diff + call-graph closure → changed vs controls.
4. **Timing** — interleaved pairs, alternating order; median, win count, spread.
5. **Adjudicate** by track.
6. **Log always**, pass or fail.

The degenerate-`nz` cases are in the matrix because the periodic-wrap peels
special-case them, and that is exactly the class of edge case hand-testing
skips.

## 5. Two tracks

| track | accuracy rule | wins by |
|---|---|---|
| **perf** | bit-identical across the whole matrix | fastest wall clock |
| **method** | answer may move | Pareto-better on (time, IntAbsDiv, DilMax) |

Two tracks are forced by the domain, not a convenience. A single bit-identity
gate would permanently exclude the largest known wins — an FFT/direct separable
pressure solve, a true recursive V-cycle, multigrid as the serial default —
because they change the answer *for the better*. And the pressure solve
currently runs a fixed sweep count with **no convergence check**, so "faster"
and "more converged" are genuinely independent axes; a change trading one for
the other is not progress.

**Bit-identical is the bar, not "close."** If output moves in the perf track,
the change is rejected until proven to be FMA contraction alone: rebuild both
sides with `-ffp-contract=off` and compare byte for byte. Anything surviving
that is a real change in the arithmetic. When output legitimately moves, every
backend the validators cross-check must move with it in the same commit.

## 6. Adjudication

Accept a **perf** record only when all hold:

- bit-identical across every configuration in the matrix, **and**
- median > 1.0 with at least ⅔ of pairs favouring the candidate, **and**
- the improvement is in a kernel the symbol diff says actually changed.

Reject when any changed kernel regresses at any grid — one regression rejects
the whole change, however large the win elsewhere. Movement in controls is
reported and ignored.

Accept a **method** record only when the answer moved *and* quality is
same-or-better on both divergence measures with at least one strictly better,
at no worse cost.

**Resolvability.** Roughly: serial kernel ~2–10%; whole-program ~13%. Below
that, say it is unresolvable rather than reporting a number. A change that is
consistent (9/9 pairs) but inside a kernel's layout band is *not* a result.

## 7. What to record

Every attempt, in `experiments/research/ledger.md` — the failures especially.
A rejected idea nobody wrote down gets retried.

A ledger entry carries: the hypothesis and mechanism, the baseline ref, whether
output was bit-identical, which symbols changed, the paired numbers with win
counts, and **where the theory was wrong**. That last field is the one that
compounds. Three examples worth the space they took:

- Splitting pass-3 divisions into their own row kernel: 1.030x at 96×48×24 but
  **0.981x at 48×24×12, 0/9 pairs**. Three `noinline` calls and six scratch
  round-trips per row cost more than the vectorisation returned on an
  11-iteration row. *Fuse rather than adding passes.*
- The first Gauss-Seidel diagonal traversal: **0.724x, 0/9 pairs.** The
  independence existed in the source and never reached the scheduler — one
  variable-trip-count loop with the peels tested inside gave the compiler
  nothing to unroll. Phasing the peels out and fixing the trip count turned the
  same traversal into 1.23x.
- Vectorising the Z-sweep stencil alone: 1.006x, 9/9 — inside the layout band,
  not resolvable.

Commit messages follow the same discipline: `perf(<area>): what changed — the
measured number`, then the mechanism, the paired numbers, and explicitly where
the theory was wrong.

## 8. Choosing what to try next

`experiments/research/backlog.md` ranks by expected value, and every entry names
the measurement motivating it so a stale one can be re-checked rather than
believed. Outcomes feed back: a rejected idea moves to "closed / do not retry"
*with its number*.

Prefer, in order: (a) a bound says there is headroom, (b) a stand-in build says
the construct is expensive, (c) the mechanism is understood well enough to
predict the size. An idea with none of the three is a guess — measure first.

For the search-strategy question in general — how to allocate effort across a
space of candidate optimizations — the autotuning literature is the reference
point, e.g. Ansel et al., *"OpenTuner: An Extensible Framework for Program
Autotuning"* (PACT 2014), which argues for ensembles of search techniques over
any single fixed strategy. Here the space is small and the hypotheses are
mechanistic, so a ranked queue beats a search; revisit that if parameter tuning
(block sizes, sweep counts) starts to dominate.

## 9. Do not

- Adjudicate against a stored baseline, or compare two numbers from different
  measurement windows.
- Report a per-kernel multi-threaded timing as a result on this machine.
- Claim a win in a kernel the symbol diff shows is byte-identical.
- Loosen a tolerance, shrink a validation grid, or regenerate a golden file to
  make a change pass. Golden files change only when the physics intentionally
  changes, in their own commit.
- Time anything from a sanitizer or `-pg` build, or compare `NAVSOLVER_PROFILE`
  totals across builds — only its per-kernel shares are comparable.
- Batch two optimizations into one commit; a mixed commit cannot be bisected
  into a win and a regression.
- Quote a hardware ceiling from a spec sheet, or trust a microbenchmark you
  have not swept for configuration.
- Pick a hypothesis without a current `characterize` run, or rank a backlog
  against a probe that reported STALE or INERT.
