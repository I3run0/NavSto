#pragma once
// =============================================================================
//  RowIndex.hpp — one (i,j) active row.
//
//  Separate from Geometry.hpp (which needs the whole SimState) so a
//  BackendConfig.hpp can declare storage for rows without a circular include.
//
//  A row spans the full k range, so row + colour derives that colour's
//  k-stride directly instead of materialising a per-cell list.
// =============================================================================

struct RowIndex { int i, j; };
