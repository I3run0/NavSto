#!/usr/bin/env python3
"""Three-dimensional renders of a NavSolver snapshot, driven by ParaView.

The VTK the solver writes is a full STRUCTURED_POINTS volume, so the 2-D
figures in scripts/plot_fields.py are slices through data that is already 3-D.
This renders what those slices leave out: vortex cores as a Q-criterion
isosurface, streamlines carrying the spanwise motion, and the wall-normal
structure the mid-plane hides.

Run with pvpython, NOT the .venv python -- it needs ParaView's own interpreter:

    pvpython scripts/render3d.py experiments/results/fig_cam10k/cam_re960_t010000.vtk \\
        --out experiments/figures --name cam_re960

The same .vtk opens directly in the ParaView GUI for interactive exploration;
this script is the reproducible path to the same views.
"""

import argparse
import os
import sys

from paraview.simple import *  # noqa: F403


def _colorbar(view, vel_range):
    """One horizontal bar along the bottom, in the same viridis the 2-D figures
    use so the two sets of images read as one field."""
    lut = GetColorTransferFunction('VelocityMagnitude')
    lut.ApplyPreset('Viridis (matplotlib)', True)
    lut.RescaleTransferFunction(*vel_range)
    bar = GetScalarBar(lut, view)
    bar.Title = 'velocity magnitude'
    bar.ComponentTitle = ''
    bar.Orientation = 'Horizontal'
    bar.WindowLocation = 'Any Location'
    bar.Position = [0.32, 0.06]
    bar.ScalarBarLength = 0.36
    bar.ScalarBarThickness = 12
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    bar.TitleFontSize = 15
    bar.LabelFontSize = 13
    return lut


def q_isosurface(src, view, out, q_level, vel_range):
    """Vortex cores: the Q-criterion is the second invariant of the velocity
    gradient, positive where rotation beats strain. The level is set from the
    field's own gradient scale rather than hard-coded, so it transfers between
    the Re-100 and Re-1000 cases."""
    grad = Gradient(Input=src)
    grad.ScalarArray = ['POINTS', 'Velocity']
    grad.ComputeQCriterion = 1
    grad.QCriterionArrayName = 'Q'
    grad.ComputeVorticity = 1
    grad.VorticityArrayName = 'Vorticity'

    contour = Contour(Input=grad)
    contour.ContourBy = ['POINTS', 'Q']
    contour.Isosurfaces = [q_level]
    contour.ComputeNormals = 1

    d = Show(contour, view)
    ColorBy(d, ('POINTS', 'VelocityMagnitude'))
    _colorbar(view, vel_range)
    d.SetScalarBarVisibility(view, True)
    return contour, grad


def streamlines(src, view, bounds, n_seed, vel_range, radius):
    """Seeded on a PLANE spanning the whole inlet face, not a line: a purely
    2-D solution then gives a flat sheet of parallel ribbons, and any spanwise
    motion shows up as ribbons leaving their z-plane."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x0 = xmin + 0.02 * (xmax - xmin)
    seed = Plane()
    seed.Origin = [x0, ymin, zmin]
    seed.Point1 = [x0, ymax, zmin]
    seed.Point2 = [x0, ymin, zmax]
    seed.XResolution = n_seed
    seed.YResolution = max(4, n_seed // 3)

    st = StreamTracerWithCustomSource(Input=src, SeedSource=seed)
    st.Vectors = ['POINTS', 'Velocity']
    st.MaximumStreamlineLength = (xmax - xmin) * 4
    st.IntegrationDirection = 'BOTH'

    tube = Tube(Input=st)
    tube.Scalars = ['POINTS', 'VelocityMagnitude']
    tube.Vectors = ['POINTS', 'Normals']
    tube.Radius = radius

    d = Show(tube, view)
    ColorBy(d, ('POINTS', 'VelocityMagnitude'))
    _colorbar(view, vel_range)
    d.SetScalarBarVisibility(view, True)
    return tube


def slice_planes(src, view, bounds, vel_range, zs):
    """Cross-stream planes: the spanwise (z) profile the mid-plane figures
    integrate away."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    out = []
    for xf in zs:
        sl = Slice(Input=src)
        sl.SliceType = 'Plane'
        sl.SliceType.Origin = [xmin + xf * (xmax - xmin), 0.5 * (ymin + ymax),
                               0.5 * (zmin + zmax)]
        sl.SliceType.Normal = [1, 0, 0]
        d = Show(sl, view)
        ColorBy(d, ('POINTS', 'VelocityMagnitude'))
        _colorbar(view, vel_range)
        d.SetScalarBarVisibility(view, True)
        out.append(sl)
    return out


def recirculation(src, view, level):
    """The reverse-flow region as a closed surface.

    Contouring streamwise velocity at a small NEGATIVE level, not at zero:
    u is exactly zero throughout the excluded solid region and on every
    no-slip wall, so a zero contour degenerates into the geometry itself.
    A negative level exists only where the flow actually reverses, so the
    surface it returns is the separation bubble alone -- the 3-D object whose
    mid-plane cut is the u = 0 line in the 2-D figures."""
    calc = Calculator(Input=src)
    calc.ResultArrayName = 'u'
    calc.Function = 'Velocity_X'

    c = Contour(Input=calc)
    c.ContourBy = ['POINTS', 'u']
    c.Isosurfaces = [level]
    c.ComputeNormals = 1

    d = Show(c, view)
    ColorBy(d, None)
    d.AmbientColor = [.10, .42, .45]
    d.DiffuseColor = [.10, .42, .45]
    d.Specular = .35
    d.Opacity = 1.0
    return c


def outline(src, view):
    o = Outline(Input=src)
    d = Show(o, view)
    d.AmbientColor = [.45, .45, .45]
    d.DiffuseColor = [.45, .45, .45]
    return o


def look(view, bounds, azimuth=1.0, elevation=0.55, zoom=1.0):
    """Three-quarter view: down the duct and slightly above it, which is the
    angle that separates a spanwise structure from a 2-D one."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraPosition = [cx - azimuth * span * 0.9,
                           cy + elevation * span * 0.8,
                           cz + span * 1.5 * zoom]
    view.CameraViewUp = [0, 1, 0]
    view.ResetCamera()
    # ResetCamera fits the bounding box with generous margins; pull in along
    # the same view direction so the duct fills the frame.
    fp = view.CameraFocalPoint
    pos = view.CameraPosition
    view.CameraPosition = [f + (p - f) * (0.72 / zoom)
                           for f, p in zip(fp, pos)]
    view.CameraParallelProjection = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('vtk')
    ap.add_argument('--out', default='experiments/figures')
    ap.add_argument('--name', default=None, help='output filename stem')
    ap.add_argument('--size', type=int, nargs=2, default=[1500, 900])
    ap.add_argument('--seeds', type=int, default=24,
                    help='inlet seed-plane resolution; raise it until recirculating '
                         'streamlines appear, lower it when ribbons merge into a sheet')
    ap.add_argument('--q', type=float, default=None,
                    help='Q-criterion level; default: 10%% of max Q')
    args = ap.parse_args()

    name = args.name or os.path.basename(args.vtk).replace('.vtk', '')
    os.makedirs(args.out, exist_ok=True)

    src = LegacyVTKReader(FileNames=[args.vtk])
    src.UpdatePipeline()
    info = src.GetDataInformation()
    bounds = info.GetBounds()
    vel_range = src.PointData['VelocityMagnitude'].GetRange(0)
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    print(f'{name}: {info.GetNumberOfPoints()} points, bounds {bounds}, '
          f'|u| in {vel_range}')

    # Q level from the data: a fixed number would pick up nothing at Re 100 and
    # everything at Re 1000.
    probe = Gradient(Input=src)
    probe.ScalarArray = ['POINTS', 'Velocity']
    probe.ComputeQCriterion = 1
    probe.QCriterionArrayName = 'Q'
    probe.UpdatePipeline()
    qrange = probe.PointData['Q'].GetRange(0)
    q_level = args.q if args.q is not None else 0.10 * qrange[1]
    print(f'  Q in {qrange}, isosurface at Q = {q_level:.4g}')
    Delete(probe)

    def new_view():
        v = CreateRenderView()
        v.ViewSize = args.size
        v.Background = [1, 1, 1]
        v.UseColorPaletteForBackground = 0
        v.OrientationAxesVisibility = 1
        v.OrientationAxesLabelColor = [0, 0, 0]
        v.CenterAxesVisibility = 0
        return v

    written = []

    # ── streamlines ──────────────────────────────────────────────────────
    v = new_view()
    streamlines(src, v, bounds, args.seeds, vel_range, radius=span * 0.0035)
    outline(src, v)
    look(v, bounds)
    p = os.path.join(args.out, f'{name}_3d_streamlines.png')
    SaveScreenshot(p, v, ImageResolution=args.size)
    written.append(p)
    Delete(v)

    # ── Q-criterion isosurface ───────────────────────────────────────────
    v = new_view()
    q_isosurface(src, v, args.out, q_level, vel_range)
    outline(src, v)
    look(v, bounds)
    p = os.path.join(args.out, f'{name}_3d_vortexcores.png')
    SaveScreenshot(p, v, ImageResolution=args.size)
    written.append(p)
    Delete(v)

    # ── recirculation bubble ─────────────────────────────────────────────
    umin = src.PointData['Velocity'].GetRange(0)[0]
    if umin < 0:
        level = 0.10 * umin
        print(f'  min u = {umin:.4g}, reverse-flow surface at u = {level:.4g}')
        v = new_view()
        recirculation(src, v, level)
        outline(src, v)
        look(v, bounds)
        p = os.path.join(args.out, f'{name}_3d_recirculation.png')
        SaveScreenshot(p, v, ImageResolution=args.size)
        written.append(p)
        Delete(v)
    else:
        print('  no reverse flow in this snapshot -- skipping bubble surface')

    # ── cross-stream slices ──────────────────────────────────────────────
    v = new_view()
    slice_planes(src, v, bounds, vel_range, [0.15, 0.35, 0.55, 0.75, 0.95])
    outline(src, v)
    look(v, bounds, azimuth=1.4, elevation=0.5)
    p = os.path.join(args.out, f'{name}_3d_slices.png')
    SaveScreenshot(p, v, ImageResolution=args.size)
    written.append(p)
    Delete(v)

    for p in written:
        print(f'  wrote {p}')


if __name__ == '__main__':
    main()
