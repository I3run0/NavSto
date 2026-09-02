#!/usr/bin/env python3
"""Figures from a NavSolver run: convergence history, mid-plane fields,
velocity profiles, and (for the Straight channel) the analytic Poiseuille check.

Reads the legacy-ASCII VTK snapshots and the convergence CSV that
src/io/VtkExporter.hpp writes, with numpy + matplotlib only -- the
experiments/notebooks copies need pandas and pyvista, which the .venv
does not carry.

Usage:
    python3 scripts/plot_fields.py experiments/results/fig_backstep \\
        --out experiments/figures
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
#  VTK legacy STRUCTURED_POINTS reader
# ---------------------------------------------------------------------------
def read_vtk(path):
    """Returns (fields, origin, spacing, dims) with fields[name] shaped
    (nz, ny, nx) for scalars and (nz, ny, nx, 3) for vectors -- the exporter
    writes i fastest, then j, then k."""
    lines = path.read_text().split("\n")
    dims = origin = spacing = None
    fields = {}
    c = 0
    while c < len(lines):
        ln = lines[c].strip()
        if ln.startswith("DIMENSIONS"):
            dims = tuple(int(v) for v in ln.split()[1:4])
        elif ln.startswith("ORIGIN"):
            origin = tuple(float(v) for v in ln.split()[1:4])
        elif ln.startswith("SPACING"):
            spacing = tuple(float(v) for v in ln.split()[1:4])
        elif ln.startswith("VECTORS") or ln.startswith("SCALARS"):
            nx, ny, nz = dims
            npts = nx * ny * nz
            name = ln.split()[1]
            comps = 3 if ln.startswith("VECTORS") else 1
            c += 1
            if lines[c].strip().startswith("LOOKUP_TABLE"):
                c += 1
            blk = "\n".join(lines[c:c + npts])
            arr = np.fromstring(blk, sep=" ")
            shape = (nz, ny, nx) if comps == 1 else (nz, ny, nx, 3)
            fields[name] = arr.reshape(shape)
            c += npts - 1
        c += 1
    return fields, origin, spacing, dims


def read_convergence(path):
    cols = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for name in rd.fieldnames:
            cols[name] = []
        for row in rd:
            for name in rd.fieldnames:
                cols[name].append(float(row[name]))
    return {k: np.asarray(v) for k, v in cols.items()}


def read_cfg(path):
    cfg = {}
    for ln in path.read_text().split("\n"):
        ln = ln.split("#")[0].strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def snapshots(run_dir, run_name):
    pat = re.compile(re.escape(run_name) + r"_t(\d+)\.vtk$")
    out = []
    for p in run_dir.glob(f"{run_name}_t*.vtk"):
        m = pat.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------
STYLE = dict(linewidth=1.4)


def _pos(a):
    """Log axes: step 0 is written before the first residual exists, so its
    row is all zeros -- mask it rather than clamping, which otherwise drags
    the y-axis down to 1e-300 and flattens the whole history."""
    return np.where(a > 0, a, np.nan)


def fig_convergence(hist, title, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    step = hist["step"]

    ax[0].semilogy(step, _pos(hist["ResidMax"]), label="max", **STYLE)
    ax[0].semilogy(step, _pos(hist["ResidRMS"]), label="RMS", **STYLE)
    ax[0].set_xlabel("time step"); ax[0].set_ylabel("momentum residual")
    ax[0].set_title("Momentum residual"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].semilogy(step, _pos(np.abs(hist["DilMax"])),
                   label=r"$\max|\nabla\cdot u|$", **STYLE)
    ax[1].semilogy(step, _pos(np.abs(hist["IntAbsDiv"])),
                   label=r"$\int|\nabla\cdot u|$", **STYLE)
    ax[1].set_xlabel("time step"); ax[1].set_ylabel("dilatation")
    ax[1].set_title("Incompressibility"); ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(step, hist["dt"], **STYLE)
    ax[2].set_xlabel("time step"); ax[2].set_ylabel(r"$\Delta t$")
    ax[2].set_title("Adaptive time step"); ax[2].grid(alpha=.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _solid_mask(mag):
    """Cells the solver never advances: outside the geometry the fields stay
    exactly zero. No-slip walls are exactly zero too, so this greys a one-cell
    wall line as well -- which is what a wall should look like."""
    return mag == 0.0


def reattachment_x(u, x, y):
    """First x downstream of the recirculation where the near-wall streamwise
    velocity turns positive again -- the reattachment point of the bubble.
    Returns None when the near-wall row never reverses."""
    j = 1                                   # first row off the bottom wall
    row = u[j, :]
    neg = np.where(row < 0)[0]
    if neg.size == 0:
        return None
    last = neg[-1]
    if last + 1 >= row.size:
        return None
    # linear interpolation across the sign change
    a, b = row[last], row[last + 1]
    return x[last] + (x[last + 1] - x[last]) * (-a / (b - a))


def fig_fields(fields, origin, spacing, title, out):
    """Mid-depth (z) plane: velocity magnitude + streamlines, pressure,
    streamwise velocity, and spanwise vorticity."""
    vel = fields["Velocity"]
    nz, ny, nx, _ = vel.shape
    kmid = nz // 2
    u = vel[kmid, :, :, 0]
    v = vel[kmid, :, :, 1]
    mag = fields["VelocityMagnitude"][kmid]
    p = fields["Pressure"][kmid]

    dx, dy = spacing[0], spacing[1]
    x = origin[0] + dx * np.arange(nx)
    y = origin[1] + dy * np.arange(ny)
    X, Y = np.meshgrid(x, y)
    solid = _solid_mask(mag)

    def grey_solid(a):
        a.contourf(X, Y, np.where(solid, 1.0, np.nan), levels=[.5, 1.5],
                   colors="0.75")

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    cf = ax[0].contourf(X, Y, np.ma.masked_where(solid, mag), levels=40,
                        cmap="viridis")
    ax[0].streamplot(x, y, u, v, color="w", density=1.6, linewidth=.5,
                     arrowsize=.6)
    fig.colorbar(cf, ax=ax[0], label=r"$|u|$")
    ax[0].set_title("Velocity magnitude with streamlines")

    cf = ax[1].contourf(X, Y, np.ma.masked_where(solid, p), levels=40,
                        cmap="RdBu_r")
    fig.colorbar(cf, ax=ax[1], label=r"$p$")
    ax[1].set_title("Pressure")

    # Centre the diverging map on u = 0: the reverse flow is a few percent of
    # the forward peak, and symmetric limits render it as flat white.
    from matplotlib.colors import TwoSlopeNorm
    umin, umax = float(u.min()), float(u.max())
    if umin < 0 < umax:
        norm = TwoSlopeNorm(vmin=umin, vcenter=0.0, vmax=umax)
        levels = np.concatenate([np.linspace(umin, 0, 20),
                                 np.linspace(0, umax, 21)[1:]])
    else:
        norm, levels = None, 40
    cf = ax[2].contourf(X, Y, np.ma.masked_where(solid, u), levels=levels,
                        cmap="RdBu_r", norm=norm)
    fig.colorbar(cf, ax=ax[2], label=r"$u$")
    ax[2].contour(X, Y, u, levels=[0.0], colors="k", linewidths=.9)
    xr = reattachment_x(u, x, y)
    sub = "black line: $u=0$"
    if xr is not None:
        ax[2].axvline(xr, color="k", ls="--", lw=.8)
        ax[2].annotate(f"reattachment $x$ = {xr:.2f}", xy=(xr, y[1]),
                       xytext=(6, 8), textcoords="offset points", fontsize=8)
        sub += f", reattachment at x = {xr:.2f}"
    ax[2].set_title(f"Streamwise velocity ({sub})")

    # Spanwise vorticity: the shear layer shed off the step edge.
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    wz = dvdx - dudy
    wlim = np.nanpercentile(np.abs(wz[~solid]), 98) if (~solid).any() else 1.0
    cf = ax[3].contourf(X, Y, np.ma.masked_where(solid, wz), levels=40,
                        cmap="PuOr_r", vmin=-wlim, vmax=wlim, extend="both")
    fig.colorbar(cf, ax=ax[3], label=r"$\omega_z$")
    ax[3].set_title(r"Spanwise vorticity $\omega_z=\partial_x v-\partial_y u$")

    for a in ax:
        grey_solid(a)
        a.set_ylabel("y")
        a.set_aspect("equal")
    ax[-1].set_xlabel("x")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return xr


def fig_profiles(fields, origin, spacing, title, out):
    vel = fields["Velocity"]
    nz, ny, nx, _ = vel.shape
    kmid = nz // 2
    u = vel[kmid, :, :, 0]
    y = origin[1] + spacing[1] * np.arange(ny)
    x = origin[0] + spacing[0] * np.arange(nx)

    stations = np.linspace(0, nx - 1, 6).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for i in stations:
        ax[0].plot(u[:, i], y, label=f"x = {x[i]:.2f}", **STYLE)
    ax[0].axvline(0, color="k", lw=.6)
    ax[0].set_xlabel("u"); ax[0].set_ylabel("y")
    ax[0].set_title("Streamwise velocity profiles"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    # Centreline and bulk flux along x -- mass conservation is visible here.
    jmid = ny // 2
    ax[1].plot(x, u[jmid, :], label="centreline u", **STYLE)
    flux = np.trapezoid(u, y, axis=0) if hasattr(np, "trapezoid") \
        else np.trapz(u, y, axis=0)
    ax[1].plot(x, flux, label=r"$\int u\,dy$ (bulk flux)", **STYLE)
    ax[1].set_xlabel("x"); ax[1].set_title("Streamwise development")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_poiseuille(fields, origin, spacing, title, out):
    """Computed outlet profile against the exact parabola the config claims:
    u(y) = 6 yN (1 - yN) Umax with yN = y / H."""
    vel = fields["Velocity"]
    nz, ny, nx, _ = vel.shape
    u = vel[nz // 2, :, :, 0]
    y = origin[1] + spacing[1] * np.arange(ny)
    H = y[-1] - y[0]
    yN = (y - y[0]) / H

    ix = int(nx * 0.75)                 # downstream, past any inlet transient
    prof = u[:, ix]
    umax = prof.max()
    exact = 6.0 * yN * (1.0 - yN) * (umax / 1.5)   # 6 yN(1-yN) Umax, peak 1.5 Umax

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(prof, y, "o", ms=3, label="computed")
    ax[0].plot(exact, y, "-", label="exact parabola", **STYLE)
    ax[0].set_xlabel("u"); ax[0].set_ylabel("y")
    ax[0].set_title(f"Profile at x index {ix} of {nx-1}")
    ax[0].legend(); ax[0].grid(alpha=.3)

    err = prof - exact
    ax[1].plot(err, y, **STYLE)
    ax[1].set_xlabel(r"$u_{\rm computed} - u_{\rm exact}$"); ax[1].set_ylabel("y")
    linf = np.max(np.abs(err))
    l2 = np.sqrt(np.mean(err ** 2))
    ax[1].set_title(rf"error: $L_\infty$={linf:.2e}, $L_2$={l2:.2e}")
    ax[1].grid(alpha=.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return linf, l2


def fig_evolution(snaps, run_name, title, out):
    """Velocity magnitude at the mid-plane for up to four snapshots."""
    picks = snaps if len(snaps) <= 4 else [snaps[0], snaps[len(snaps) // 3],
                                           snaps[2 * len(snaps) // 3], snaps[-1]]
    fig, ax = plt.subplots(len(picks), 1, figsize=(10, 2.3 * len(picks)),
                           sharex=True)
    ax = np.atleast_1d(ax)
    vmax = None
    for a, (step, path) in zip(ax, picks):
        f, origin, spacing, _ = read_vtk(path)
        mag = f["VelocityMagnitude"][f["VelocityMagnitude"].shape[0] // 2]
        ny, nx = mag.shape
        x = origin[0] + spacing[0] * np.arange(nx)
        y = origin[1] + spacing[1] * np.arange(ny)
        if vmax is None:
            vmax = mag.max()
        solid = _solid_mask(mag)
        cf = a.contourf(x, y, np.ma.masked_where(solid, mag), levels=30,
                        cmap="viridis", vmin=0, vmax=vmax)
        a.contourf(x, y, np.where(solid, 1.0, np.nan), levels=[.5, 1.5],
                   colors="0.75")
        fig.colorbar(cf, ax=a, label=r"$|u|$")
        a.set_ylabel("y"); a.set_aspect("equal")
        a.set_title(f"step {step}", fontsize=9)
    ax[-1].set_xlabel("x")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("experiments/figures"))
    ap.add_argument("--name", default=None, help="runName (inferred if omitted)")
    args = ap.parse_args()

    run_dir = args.run_dir
    name = args.name
    if name is None:
        csvs = list(run_dir.glob("*_convergence.csv"))
        if not csvs:
            sys.exit(f"no *_convergence.csv in {run_dir}")
        name = csvs[0].name[: -len("_convergence.csv")]

    args.out.mkdir(parents=True, exist_ok=True)
    snaps = snapshots(run_dir, name)
    if not snaps:
        sys.exit(f"no {name}_t*.vtk snapshots in {run_dir}")

    cfg_files = list(run_dir.glob("*.cfg"))
    cfg = read_cfg(cfg_files[0]) if cfg_files else {}
    shape = cfg.get("geometryShape", "?")
    re_num = cfg.get("reynoldsNumber", "?")
    grid = f"{cfg.get('numCellsX','?')}x{cfg.get('numCellsY','?')}x{cfg.get('numCellsZ','?')}"
    head = f"{name} — {shape}, Re={re_num}, {grid}"

    hist = read_convergence(run_dir / f"{name}_convergence.csv")
    fig_convergence(hist, f"{head} — convergence", args.out / f"{name}_convergence.png")

    last_step, last_path = snaps[-1]
    fields, origin, spacing, _ = read_vtk(last_path)
    xr = fig_fields(fields, origin, spacing, f"{head} — step {last_step}",
                    args.out / f"{name}_fields.png")
    fig_profiles(fields, origin, spacing, f"{head} — step {last_step}",
                 args.out / f"{name}_profiles.png")
    if len(snaps) > 1:
        fig_evolution(snaps, name, f"{head} — evolution",
                      args.out / f"{name}_evolution.png")

    msg = ""
    if shape == "Straight":
        linf, l2 = fig_poiseuille(fields, origin, spacing,
                                  f"{head} — analytic check",
                                  args.out / f"{name}_poiseuille.png")
        msg = f"  Poiseuille error: Linf={linf:.3e}  L2={l2:.3e}"

    print(f"{name}: {len(snaps)} snapshots, last step {last_step}, "
          f"final ResidMax={hist['ResidMax'][-1]:.3e}, "
          f"DilMax={hist['DilMax'][-1]:.3e}"
          + (f", reattachment x={xr:.3f}" if xr is not None else ""))
    if msg:
        print(msg)
    for p in sorted(args.out.glob(f"{name}_*.png")):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
