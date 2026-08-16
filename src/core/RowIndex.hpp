#pragma once
// =============================================================================
//  RowIndex.hpp — one (i,j) active row, spanning the full k range.
//
//  Split out of Geometry.hpp (which needs all of SimState) so a
//  BackendConfig.hpp can declare row storage without a circular include.
// =============================================================================

struct RowIndex { int i, j; };
