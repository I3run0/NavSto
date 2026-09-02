# Characterization — 34c3f0c on Intel(R) Core(TM) 7 150U
_2026-09-02 02:58Z_

## Workload
- grid 180x60x30; 124,530 active of 361,088 ghosted cells (34.5%)
- one fp64 field 2.9 MB; six-field working set 17.3 MB; L2 7.9 MB, L3 12.6 MB
- regime: **STREAMING (the working set exceeds L3)** — a win measured in one regime need not transfer to the other
- RK4Transient: 5 computeAccelerations and 4 solvePressurePoisson per step, each pressure call doing 10 sweeps

## Attribution
| kernel | share | ms/call |
|---|---|---|
| computeAccelerations | 61.1% | 5.009 |
| solvePressurePoisson | 25.7% | 2.635 |
| buildPressureSource | 6.3% | 0.650 |
| updateVelocities | 4.8% | 0.496 |
| computeDivergence | 0.8% | 0.344 |
| computeMomentumResidual | 0.8% | 0.326 |
| adaptTimeStep | 0.4% | 0.157 |

## Working-set sweep — bandwidth-bound or not
ns per active cell as the working set grows past the caches. A bandwidth-bound kernel falls off a cliff; a latency- or throughput-bound one barely moves.

| grid | 6 fields | adaptTimeStep | buildPressureSource | computeAccelerations |
|---|---|---|---|---|
| 48x24x12 | 1 MB | 0.7 | 4.5 | 21.7 |
| 96x48x24 | 6 MB | 0.7 | 3.9 | 23.0 |
| 144x72x36 | 20 MB | 0.7 | 3.5 | 22.4 |
| 192x96x48 | 46 MB | 0.9 | 3.5 | 26.1 |

## Probes — what would speeding each construct up be worth
Same-shape stand-ins, deliberately wrong results, timed interleaved against the real build. Run at both Reynolds numbers the project uses: they take different branches.

| construct | kernel | Re=10000 | Re=100 | asks |
|---|---|---|---|---|
| exp | computeAccelerations | 0.2% | 4.6% | cost of the transcendental in the weight evaluation |
| divisions | computeAccelerations | 21.0% | 21.2% | cost of every double division on the path |
| branch-chain | computeAccelerations | 22.9% | 24.7% | cost of the four-way branch selection, exp included |
| gs-recurrence | solvePressurePoisson | 26.9% | 30.4% | cost of the Gauss-Seidel dependent chain in k |

## Reading this
- The dominant kernel is **computeAccelerations**. Rank the backlog against the probe shares above, not against intuition.
- A probe share is an upper bound on what removing that construct buys, and removing it is usually not legal — the value is knowing which constructs are worth restructuring *around*.
- Probe shares do not sum: removing one division lets the others' latency overlap, so each measured alone looks larger than its marginal share.
