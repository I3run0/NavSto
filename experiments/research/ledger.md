# Research ledger

Append-only. Negative results belong here too — a rejected idea nobody wrote down gets retried.

## 2026-09-02 03:33Z — RECORD (perf)
- **Attempt:** row-level branch dispatch in the weight operand passes
- **Against:** `HEAD`   **Backend:** serial
- **Bit-identical:** True
- **Code changed in:** computeAccelerations
- **Timing:** 1.016x, 3/5 pairs; 27.67 s -> 27.34 s
- **Quality:** IntAbsDiv 0.7699 -> 0.7699, DilMax 76.50 -> 76.50
- **Verdict:** 1.016x (3/5 pairs), answer unchanged

## 2026-09-02 03:42Z — REJECTED (perf)
- **Attempt:** row-level branch dispatch in the weight operand passes
- **Against:** `HEAD`   **Backend:** serial
- **Bit-identical:** True
- **Code changed in:** computeAccelerations
- **Timing:** 1.014x, 7/9 pairs; 27.23 s -> 26.41 s
- **Quality:** IntAbsDiv 0.7699 -> 0.7699, DilMax 76.50 -> 76.50
- **Verdict:** not resolvably faster: 1.014x, 7/9 pairs, sign-test p=0.090 (need p<=0.05)

## 2026-09-02 03:45Z — note (f883ac1)
REJECTED — row-level branch dispatch in the weight operand passes.

Hypothesis: the branch chain is 22.9-24.7% of computeAccelerations while exp is
only 0.2-4.6% of it, so most of that cost is the per-cell branching itself. DPe
varies smoothly along a k-row, so classify the row once and run a branch-free
filler; at Re=10000 every cell sits in the |DPe|>200 tail and needs no exp at
all, which vectorises completely.

Measured, paired and interleaved, bit-identical output across all 10 configurations:
  whole-program target   1.014x, 7/9 pairs, sign-test p=0.090   (not significant)
  kbench 96x48x24        0.962x, 1/9 pairs
  kbench 144x72x36       0.909x, 0/9 pairs
  144x72x36, with the classification fused into the DPe pass to make it free:
                         0.872x, 0/9 pairs  -- worse still

Two things learned, both worth not retrying:

The 23% is not the branching. It is the work inside the branches -- storing five
or six operand rows per cell, and exp where that branch is taken. Removing the
tests recovers almost none of it, and the classification pass is pure added cost
on any row that turns out to be Mixed. Rows near a wall have DPe -> 0 and fall
into the polynomial branch, so Mixed is common in a real flow.

Fusing the min/max reduction into the DPe loop to make classification free made
it WORSE, not better: four reduction accumulators put a dependency chain into a
loop that had been cleanly vectorised, costing more than the separate scan did.

Also note the whole-program target and the kernel bench disagreed in SIGN here
(1.014x vs 0.909x). The target dilutes a single-kernel change to ~61% and its
synthetic-vs-evolved field changes which branch rows take. A single-kernel
hypothesis must be adjudicated at the kernel, not only on the record target.

## 2026-09-02 04:04Z — RECORD (perf)
- **Attempt:** hoist the loop-invariant /localRe into a Markstein-refined reciprocal
- **Against:** `HEAD`   **Backend:** serial
- **Bit-identical:** True
- **Code changed in:** computeAccelerations
- **Timing:** 1.029x, 9/9 pairs; 37.58 s -> 35.89 s
- **Quality:** IntAbsDiv 0.7699 -> 0.7699, DilMax 76.50 -> 76.50
- **Verdict:** 1.029x, 9/9 pairs, p=0.002, answer unchanged

## 2026-09-02 04:06Z — note (2013d4d)
RECORD — refined-reciprocal for the invariant divisor (cfec776).

Kernel-level, paired and interleaved, 9 pairs each:
  96x48x24    computeAccelerations 1.148x  9/9 (min 1.135)
  144x72x36   computeAccelerations 1.114x  9/9 (min 1.066)
Whole-program target 1.029x, 9/9, p=0.002 -- lower because the kernel is 61%
of the step and the machine was throttling hard (the same record-holder binary
measured 27.2 s earlier in the session and 37.6 s during this attempt).

Bit-identical across all 10 configurations including the multigrid path.
solvePressurePoisson is a control and reads 1.001x / 1.015x, as it should.

Method note worth keeping: the idea was on the backlog as BLOCKED by the
accuracy gate, with an operation-count estimate of ~6%. Pricing it as a
diagnostic first -- build the illegal version, measure it, throw it away --
returned 11.2-22.8%, and that is what justified hunting for a legal route
rather than arguing about the rule. Markstein turned out to give the
correctly-rounded quotient, so no rule had to move at all. Price a blocked
idea before debating the gate that blocks it.

## 2026-09-02 — UNVALIDATED LEAD (not a record)

Gauss-Seidel block size may want re-tuning to 7. Single-shot kbench, taken as
the battery collapsed from 22% to 5%, so this is an observation and NOT a
measurement — no repeats, no interleaving, no pairing:

  GSB=4  1.209 / 3.199 ms   (96x48x24 / 144x72x36)
  GSB=5  0.922 / 2.781
  GSB=6  0.915 / 2.798   <- committed value
  GSB=7  0.873 / 2.680   <- apparent best, ~5% under 6 at both grids
  GSB=8  1.010 / 2.988

Reverted to the committed GSB=6 rather than land an unvalidated change. The
plausible mechanism is that 140a9a9 tuned this before cfec776 changed the
binary's layout and cache footprint, so the optimum may genuinely have moved --
which would also mean the tuning is layout-sensitive and needs re-checking after
any sizeable change to the accel kernels, not just this once.

To settle it, on mains power:
  python3 scripts/autoresearch.py attempt -m "gauss-seidel block size 7" --repeats 9
plus a kernel-level paired run at both grids, since this is a single-kernel
change and the record target dilutes it (see the 2026-09-02 branch-dispatch
entry, where target and kbench disagreed in sign).

## 2026-09-02 15:18Z — REJECTED (perf)
- **Attempt:** gauss-seidel diagonal block size 7, re-tuned after cfec776 changed the binary layout
- **Against:** `HEAD`   **Backend:** serial
- **Bit-identical:** True
- **Code changed in:** solvePressurePoisson
- **Timing:** 1.018x, 5/9 pairs; 19.86 s -> 19.80 s
- **Quality:** IntAbsDiv 0.7699 -> 0.7699, DilMax 76.50 -> 76.50
- **Verdict:** not resolvably faster: 1.018x, 5/9 pairs, sign-test p=0.500 (need p<=0.05)

## 2026-09-02 15:32Z — REJECTED (perf)
- **Attempt:** gauss-seidel diagonal block size 7
- **Against:** `HEAD`   **Backend:** serial
- **Bit-identical:** True
- **Code changed in:** solvePressurePoisson
- **Timing:** 0.902x, 2/10 pairs; 26.91 s -> 27.08 s
- **Quality:** IntAbsDiv 0.7699 -> 0.7699, DilMax 76.50 -> 76.50
- **Verdict:** not resolvably faster: 0.902x, 2/10 pairs, sign-test p=0.989 (need p<=0.05)

## 2026-09-02 15:34Z — note (4ffcf5d)
REJECTED — Gauss-Seidel diagonal block size 7 (settled).

The lead from the low-battery sweep does not survive proper measurement. With
adaptive paired sampling:
  kbench 96x48x24    solvePressurePoisson 0.990x  9/19 pairs  p=0.676
  kbench 144x72x36   solvePressurePoisson 1.031x  12/18 pairs p=0.119
  whole-program      0.902x  2/10 pairs  p=0.989 (sampler stopped early: even
                     winning every remaining pair could not have reached 0.05)
Bit-identical, as expected -- block size changes the traversal, not the
arithmetic. GSB stays at 6.

Four earlier fixed-count runs gave 1.027x (8/9), 0.997x (4/9), 1.052x (8/11)
and 1.052x (7/11) -- every median positive, none significant, and the apparent
5% from the single-shot sweep was noise throughout. This is what a fixed pair
count cannot do: it neither confirms nor kills a marginal hypothesis, so the
same question gets re-litigated. Adaptive sampling closed it in one run by
spending 18-19 pairs where the answer was in doubt and bailing after 10 where
it was hopeless.

Also the first exercise of kernel-level adjudication, and it mattered: the
whole-program target read 0.902x while the kernel read 1.031x at one grid. A
pressure-solve change is ~20% of the step, so the target cannot see it either
way.

## 2026-09-02 15:58Z — RECORD (perf)
- **Attempt:** interleave each pressure block's k=nZ phase with the next block's k=1 phase
- **Against:** `HEAD`   **Backend:** serial
- **Bit-identical:** True
- **Code changed in:** solvePressurePoisson
- **Timing:** 1.068x, 10/18 pairs; 21.95 s -> 22.32 s
- **Quality:** IntAbsDiv 0.7699 -> 0.7699, DilMax 76.50 -> 76.50
- **Verdict:** 1.068x target (10/18, p=0.407); decisive at the kernel; answer unchanged

## 2026-09-02 15:59Z — note (2e104ff)
REJECTED — interleaving each pressure block's k=nZ phase with the next block's k=1 phase.

The independence argument held: bit-identical across all 10 configurations, so
the two phases really do not read what the other writes, in either BC. The
speed did not follow.

  kbench 96x48x24    solvePressurePoisson 1.011x  15/21  p=0.039
  kbench 144x72x36   solvePressurePoisson 1.004x  11/21  p=0.500
  whole-program      1.068x  10/18  p=0.407, spread 0.597-2.089 (unusable)

Rejected under the Bonferroni correction added in the same commit: three tests
were run, so the threshold is 0.05/3 = 0.0167 and the best observed p is 0.039.
Without the correction the harness called this a RECORD on a 1.1% effect that
was significant at one grid out of two.

Why so little. The estimate was ~10% of the pressure solve, on the reasoning
that the two peel phases are ~a fifth of a block's time and interleaving halves
their latency. Either they are a smaller share than that, or the out-of-order
window was already covering them across the block boundary without help -- the
diagonal interior that precedes the phases has GSB cells in flight, so the
machine has plenty of independent work queued when the phase chain starts.
Latency that is already hidden cannot be hidden twice.

Do not retry without first measuring the phases' actual share; the ~21%
figure was reasoned, never measured.
