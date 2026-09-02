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
