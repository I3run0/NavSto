# Hypothesis backlog

Ranked by measured expected value, not by appeal. Every entry names the
measurement that motivates it, so a stale one can be re-checked rather than
believed. Move an entry to the ledger once `autoresearch.py verify` has ruled
on it, with the number it actually produced.

## Where the time is now (attribution, 96x48x24 serial)

| kernel | share | state |
|---|---|---|
| computeAccelerations | ~59% | at the `vdivpd` floor; divisions 29-33%, operand pass 21-26% |
| solvePressurePoisson | ~19% | recurrence down from 69-75% to 24-31% after diagonal blocking |
| buildPressureSource | ~9% | vectorised (2.26x, earlier work) |
| everything else | ~13% | row-kernel vectorised |

Production reality check: 360x120x60, RK4, 200k steps is ~12.9 h serial,
~10.2 h CUDA fp64, ~3.4-5.2 h CUDA fp32.

## 1. FFT / direct separable pressure solve  — largest available win

The geometry is z-invariant (`iLow/iHigh` indexed by j only, `jLow/jHigh` by i
only), z is periodic, dz uniform. The Poisson operator is therefore separable
in z: a real-to-complex FFT in z decouples it into nZ/2+1 independent 2-D
Helmholtz problems on the same cross-section, and that cross-section operator
is constant in time — factorise once, reuse every step. Exact to round-off,
O(N log N), and it deletes `numPressureIter` as a tuning knob.

Why this matters beyond speed: **the pressure solve currently never checks
convergence.** It runs a fixed sweep count and returns, which is why DilMax
sits at 18-76 instead of ~1e-12. The projection is incomplete, so the velocity
field is not actually divergence-free. This is an accuracy fix that happens to
also be faster.

Caveats: needs FFTW (host) / cuFFT (device); the k_z=0 mode keeps the
pure-Neumann singularity and the existing null-space pinning; golden files
move, so this is an intentional-physics-change commit under CLAUDE.md, not a
perf commit. Breaks if the geometry ever becomes z-dependent.

## 2. Multigrid for the parallel backends

`pressureSolver` is honoured **only** by the serial backend; OpenMP and CUDA
ignore it and hardcode red-black SOR. Measured (400 steps, 180x60x30,
production physics, best of 3):

| config | time | IntAbsDiv | DilMax |
|---|---|---|---|
| serial GS it=10 (default) | 19.45s | 0.770 | 76.5 |
| serial MG 2/16/2 | 20.69s | 0.621 | 18.2 |
| omp6 RB it=7 | 26.87s | 0.630 | 43.3 |
| cuda32 RB it=20 | 16.90s | 0.450 | 17.9 |

Multigrid and red-black are complementary: MG crushes the *max* divergence
(smooth modes) and saturates near IntAbsDiv 0.55; red-black grinds the *mean*
down further. Multigrid with a red-black smoother on the GPU is the
combination nobody has, and its expensive part is exactly what red-black
parallelises.

## 3. Flip the serial default to Multigrid  — nearly free

Same table: MG 1/8/1 costs 19.82s against GS it=10's 19.45s and gives 2.8x
lower DilMax (27.5 vs 76.5). GS also plateaus — past 10 sweeps IntAbsDiv stops
improving (0.770 -> 0.789) while cost keeps rising. Changes results, so it
needs the golden-file treatment.

## 4. Residual-based stopping criterion

Today `numPressureIter` is a quality knob with no guarantee attached. A
residual test would spend iterations where they are needed and make the
quality claim checkable. Cheap; independent of items 1-3.

## 5. GPU has 2-3x of headroom; the CPU has much less

Measured device bandwidth 90.4 GB/s float / 90.0 GB/s double (94% of the
MX570's 96 GB/s), against 23.5 GB/s CPU single-thread and 32.2 GB/s all-thread.
fp64 costs 1.94-2.20x on the device — almost exactly the byte ratio, not the
1/64 fp64 ALU ratio, so the device kernels are traffic-limited, not ALU-limited.
Effort spent device-side should pay better than more CPU micro-optimisation.

Do NOT trust the roofline efficiency percentages until traffic is measured:
the hand count is uncertain by ~5x depending on whether it uses the full
ghosted array or active cells only (19.5% of it), `ncu` cannot read counters
under WSL2, and `perf` lacks kernel support here.

## 6. Interleave the pressure blocks' k=1 and k=nZ phases

Within `blockRows`, those two phases are six-cell dependent chains along j and
cannot be broken in lexicographic order — but block B's k=nZ phase and block
B+1's k=1 phase are mutually independent and could run interleaved for two
chains. Worth ~4-5% of the whole program. Superseded if item 1 lands.

## 7. Row-level branch dispatch in the operand pass

The scalar operand pass is 21-26% of computeAccelerations while `exp` alone is
<=7% — most of it is branching and stores. When every cell in a row takes the
same branch of `computeExponentialWeights` the whole row could go through a
vectorised path. At Re=10000 all cells take `DPe > 200` and never call `exp`.
Buys nothing at Re=100, which is what the reward and roofline workloads run,
so its value depends on which Re production actually cares about.

## Closed / do not retry

- **Vectorising the Z sweep's stencil arithmetic alone**: 1.006x, 9/9 pairs —
  inside the 0.992-1.013x layout band, not resolvable. The arithmetic beside
  the divisions was never the constraint.
- **Splitting pass-3 divisions into their own row kernel**: 0.981x at
  48x24x12, 0/9 pairs. Three noinline calls and six scratch round-trips per row
  cost more than the vectorisation returned on an 11-iteration row. Fuse
  instead of adding passes.
- **Reducing the division count**: fixed by the scheme. Would change the
  arithmetic, which the gate forbids.
