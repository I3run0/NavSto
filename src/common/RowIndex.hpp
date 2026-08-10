#pragma once
// =============================================================================
//  RowIndex.hpp — one (i,j) active-row reference.
//
//  Its own header, separate from Geometry.hpp (which builds lists of these and
//  therefore needs the whole SimState), so a backend's BackendConfig.hpp can
//  declare storage for them without including SimState -- which includes
//  BackendConfig.hpp in turn.
//
//  A row covers the FULL k span [1,numCellsZ] -- k is never masked by the
//  active-domain geometry -- so a row plus a color is enough to derive that
//  color's k-stride directly (kStart = ((i+j)%2==0) ? 2 : 1 for red, the
//  opposite for black), instead of materializing a per-cell index list. See
//  Geometry.hpp for why a per-cell list is a gather that defeats prefetching.
// =============================================================================

struct RowIndex { int i, j; };
