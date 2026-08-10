# X-sweep loop-order restructuring

**Date:** 2026-08-07
**Builds on:** `docs/roofline.md` (verdict: proceed), `docs/serial-optimization.md`
(originally identified this, deferred pending the roofline verdict)

## What changed

`computeAccelerations`'s X-direction sweep looped `for(j) for(k) for(i){5 passes}`
— `i` innermost, stride `sJ*sK` in `GridField`'s storage (which has `k` as
the fastest-varying/unit-stride index). `i` genuinely can't move to
innermost: each of the 5 passes over `i` depends on the previous pass's
results at neighboring `i` (a real recurrence). But `k` has no such
coupling — nothing in the X-sweep body references `k±1`, only `i±1` — so it
was free to become the innermost, unit-stride loop instead of a fixed outer
index.

Restructured to `for(j) for(pass 1..5, same order as before) for(i) for(k)`.
Same math, same recurrence order along `i`, only which loop is innermost
changed. This needed the 5 passes' temporary scratch buffers (`ppie`,
`ppiw`, `qsie`, `Ku`, `Kv`, `Kw`) to hold a value per `(i,k)` pair
simultaneously instead of being overwritten per `k` — widened from 1-D to
2-D (new `ppieXK`/`ppiwXK`/`qsieXK`/`KuXK`/`KvXK`/`KwXK` in `SimState`,
flattened as `buf[(i+1)*scratchKLen + (k-1)]`).

**Only the X-sweep was touched.** The Y-sweep has the identical problem
(`j` innermost, also not matching storage) and is flagged as the natural
next step, not yet done — see Status below. The Z-sweep already has `k`
innermost and was correct already; left untouched.

**Deliberately separate buffers, not a shared 2-D upgrade of the original
`ppie`/`ppiw`/`qsie`/`Ku`/`Kv`/`Kw`.** Those are still used, unmodified and
1-D, by the (not yet restructured) Y-sweep and the Z-sweep — `Ku`/`Kv`/`Kw`
specifically are shared/reused across all three direction sweeps in
sequence within one `computeAccelerations` call. Giving the X-sweep its own
dedicated 2-D buffers (`KuXK` etc.) avoided having to redesign that sharing
across all three sweeps at once, matching the "one sweep at a time" plan.
The old 1-D `ppie`/`ppiw`/`qsie` (X-sweep-specific names, not shared with
Y/Z) became fully dead after this change and were removed from `SimState`.

## Verification

`scripts/validate.py`'s existing checks (conservation blow-up detector,
analytical Poiseuille check) aren't sufficient alone for this kind of
change — the Poiseuille check seeds the *exact* solution as initial
condition, so residuals are already ~0 there regardless of whether this
restructuring introduced a subtle bug away from equilibrium.

Added `tests/backend/GoldenFieldTests.cpp`: captures `accelX`/`accelY`/`accelZ`
field-by-field (not just scalar residuals, which can hide a compensating
error) from the pre-restructuring implementation on a deliberately
**non-equilibrium** config (`AbruptExpansion` + `InletProfile` init, 24×12×6
— chosen over `PotentialFlow` init for denser non-zero coverage, ~10% of
cells vs. ~1%), stored as `tests/backend/golden/accel_abrupt_expansion_24x12x6.txt`.
Compares every cell within `1e-12` relative tolerance (`1e-14` absolute
floor for near-zero values). **Passes** against the restructured
implementation — bit-for-bit equivalent within floating-point noise.

Additionally verified: `make test` (15/15, including the new golden test),
`scripts/validate.py` (all pass, and every `ResidMax`/`DilMax` value is
**identical** to the pre-restructuring run — not just "close", exactly
matching, across both the `AbruptExpansion` and `RoundedCorner+Periodic`
conservation-check configs), and AddressSanitizer clean across 5 different
grid shapes/aspect-ratios plus the periodic-lateral-BC code path (which
changes the X-sweep's `KKfim` upper bound from `numCellsZ-1` to `numCellsZ`,
exercising the 2-D scratch buffer's `k`-dimension boundary).

## Measured performance effect

Clean `-O3` builds (never the `-pg` build), min of 5 repeats, before vs.
after commit, `AbruptExpansion`, 50 steps:

| Grid | before (s) | after (s) | speedup |
|---|---|---|---|
| tiny (24×12×6, ~1.6k cells) | 0.0113 | 0.0111 | 1.02x |
| small (48×24×12, ~13k cells) | 0.0809 | 0.0772 | 1.05x |
| medium (96×48×24, ~106k cells) | 0.7535 | 0.6756 | 1.12x |
| large (144×72×36, ~358k cells) | 2.7144 | 2.5239 | 1.08x |

Whole-program speedup is modest (2-12%) — expected, since only one of
three direction sweeps changed, and `computeAccelerations` itself is
roughly a third to two-fifths of total wall time (see `docs/roofline.md`).
Isolated via `gprof` on the medium grid, 100 steps: `computeAccelerations`'s
own self-time dropped **0.35s → 0.29s (~17%)** from this change alone —
consistent with fixing roughly a third of the function's cost by a larger
factor. The speedup grows with grid size (1.02x→1.12x tiny→medium), as
expected for a cache-locality fix — a bigger working set has more to lose
from cache-unfriendly access, though it doesn't hold perfectly monotonic
out to "large" (1.08x), within the range of normal run-to-run noise.

## Status: Y-sweep not yet done

Per the approved plan, this was scoped as "one sweep at a time, verify,
then the next" — X-sweep is done and verified; the Y-sweep (identical
`j`-innermost mismatch) is the natural next step, using the same technique
(widen its scratch buffers, invert to put `k` innermost) and the same
golden-field verification approach (a new golden capture, or extending the
existing one, from the state *after* this X-sweep change).
