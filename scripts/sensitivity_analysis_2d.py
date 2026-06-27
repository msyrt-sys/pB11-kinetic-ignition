#!/usr/bin/env python3
"""
sensitivity_analysis_2d.py — 2D uncertainty-propagation map for p-11B kinetic ignition.

Central verification/sensitivity figure for the manuscript (addresses PoP referee
#2.10 uncertainty propagation and #2.11 traceability). Sweeps the KINETIC-PEAK
fusion/bremsstrahlung ratio P_F/P_B over the two dominant modeling uncertainties:

  x-axis : n_alpha/n_e  in [0.01, 0.10]
           (steady-state alpha-ash density; absorbs the tau_s ash-confinement
            uncertainty. Ochs active-removal ~0.01 ; Putvinski Fig.B1 ~0.10.)
  y-axis : Belloni F-factor scale in [0.5x, 2x] of nominal
           (the factor-2 uncertainty of the R-matrix elastic alpha-p cross
            section vs SigmaCalc 2.0, flagged by referee #2.4.)

for BOTH the Tentori-Belloni 2023 ('TB') and Wang 2026 ('wang') fusion cross
sections (selected via cross_sections.CROSS_SECTION).

KEY STRUCTURAL RESULT (verified at runtime, --verify-collapse):
  The alpha->proton diffusion is  D_alpha ~ (n_alpha/n_e) x F_scale, so the
  kinetic enhancement depends ONLY on the product  p = (n_alpha/n_e) x F_scale.
  The two uncertainties are DEGENERATE and the ignition boundary P_F/P_B = 1 is
  a hyperbola p = p*. This is exploited for speed (enh computed on a 1D p-grid)
  and is itself a reportable finding.

This script imports the unmodified physics modules and injects the F-factor
scale by wrapping alpha_source.belloni_2021_factor at runtime (no on-disk
change to the physics modules).

Outputs (to --out, default ../figures):
  - sensitivity_2d.png             : 2x2 figure (contours, ignition map, 1D collapse)
  - sensitivity_2d_grid.csv        : raw per-cell data (both cross sections)
  - sensitivity_2d_summary.csv     : per-cross-section summary (p*, % sub-ignition, ...)

Usage:
  python sensitivity_analysis_2d.py
  python sensitivity_analysis_2d.py --na-points 40 --fs-points 40 --verify-collapse
  python sensitivity_analysis_2d.py --cross-sections wang --t-min 140 --t-max 420
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

# ------------------------------------------------------------------
# Import the (unmodified) physics modules; silence their import-time
# self-test prints (alpha_p_phaseshift / belloni_full_implementation emit
# diagnostic tables at module import).
# ------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_THIS, '..', 'src'))
sys.path.insert(0, _SRC)
sys.path.insert(0, _THIS)

_real_stdout = sys.stdout
sys.stdout = io.StringIO()
import cross_sections as cs            # noqa: E402
import power_balance as pb            # noqa: E402
import alpha_source as als            # noqa: E402
import main_validation as mv          # noqa: E402
import belloni_full_implementation    # noqa: E402,F401  (trigger import-time prints here)
import alpha_p_phaseshift             # noqa: E402,F401
sys.stdout = _real_stdout

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

# ------------------------------------------------------------------
# Runtime injection of the Belloni F-factor scale (no on-disk edit).
# ------------------------------------------------------------------
_orig_belloni = als.belloni_2021_factor
_F_SCALE = [1.0]


def _scaled_belloni(v_proton_cm_s, T_p_keV):
    return _F_SCALE[0] * _orig_belloni(v_proton_cm_s, T_p_keV)


als.belloni_2021_factor = _scaled_belloni


# ------------------------------------------------------------------
# Core kernels
# ------------------------------------------------------------------
def fp_enh(T_i, n_p, n_B, n_alpha_over_ne, f_scale):
    """Kinetic enhancement enh = P_F[f_FP] / P_F[Maxwell] at one operating point."""
    _F_SCALE[0] = f_scale
    T_e = pb.find_self_consistent_Te(n_p, n_B, T_i)
    r = mv.fp_coupled_calculation(T_i, T_e, n_p, n_B, include_alpha=True,
                                  n_alpha_over_ne=n_alpha_over_ne,
                                  include_belloni=True)
    return r['kinetic_enhancement']


def thermal_ratio_curve(sigma_name, T_grid, n_p, n_B):
    """Thermal (no kinetic) P_F/P_B(T) for the chosen cross section."""
    cs.CROSS_SECTION = sigma_name
    return np.array([pb.ignition_check(n_p, n_B, T)['P_F_over_P_Brem']
                     for T in T_grid])


def enh_vs_p(sigma_name, T_grid, p_grid, n_p, n_B):
    """enh(T; p) on a (T x p) grid, exploiting the p = n_alpha x F_scale collapse.

    enh depends only on the product p, so we evaluate with n_alpha_over_ne = p,
    f_scale = 1.  burnout (nu_fus ~ sigma) is cross-section dependent, so this is
    computed once per cross section.
    """
    cs.CROSS_SECTION = sigma_name
    out = np.zeros((len(T_grid), len(p_grid)))
    for j, p in enumerate(p_grid):
        for i, T in enumerate(T_grid):
            out[i, j] = fp_enh(T, n_p, n_B, n_alpha_over_ne=p, f_scale=1.0)
    return out


def verify_collapse(n_p, n_B, T_probe=180.0):
    """Confirm enh(n_alpha, F) == enh(n_alpha*F, 1) for a few off-diagonal points."""
    checks = [
        [(0.05, 1.0), (0.10, 0.5), (0.025, 2.0)],   # p = 0.05
        [(0.02, 1.0), (0.01, 2.0)],                 # p = 0.02
    ]
    cs.CROSS_SECTION = 'wang'
    ok = True
    print("  [verify-collapse] enh(180 keV) for equal-p (n_alpha, F_scale) pairs:")
    for grp in checks:
        vals = [fp_enh(T_probe, n_p, n_B, na, fs) for (na, fs) in grp]
        spread = max(vals) - min(vals)
        ok = ok and spread < 1e-6
        p = grp[0][0] * grp[0][1]
        print(f"    p={p:.3f}: " + ", ".join(f"{v:.4f}" for v in vals)
              + f"   spread={spread:.1e}")
    print(f"  [verify-collapse] {'PASS' if ok else 'FAIL'} "
          f"(enh is a function of p = n_alpha x F_scale alone)")
    return ok


# ------------------------------------------------------------------
# Map assembly
# ------------------------------------------------------------------
def build_maps(args):
    n_i = args.n_i
    n_p = (1.0 - args.f_B) * n_i
    n_B = args.f_B * n_i

    T_grid = np.linspace(args.t_min, args.t_max, args.t_points)
    na_axis = np.linspace(args.na_min, args.na_max, args.na_points)
    fs_axis = np.linspace(args.fs_min, args.fs_max, args.fs_points)

    # 1D p-grid spanning the box corners, log-spaced (enh varies smoothly in p).
    # Handle fs_min=0 (Belloni-off floor): keep an explicit 0.0 node (= no alpha
    # coupling) and start the log grid at a small positive floor.
    p_hi = args.na_max * args.fs_max
    p_lo = args.na_min * args.fs_min
    p_floor = p_lo if p_lo > 0 else p_hi * 1e-3
    p_grid = np.unique(np.concatenate([
        [0.0],  # F_scale=0 -> no alpha-p coupling at all (absolute floor)
        np.geomspace(p_floor, p_hi, args.p_points),
    ]))

    NA, FS = np.meshgrid(na_axis, fs_axis)   # shape (fs, na)
    P = NA * FS                              # product per cell

    results = {}
    for sigma in args.cross_sections:
        print(f"  computing enh(T; p) for cross section '{sigma}' "
              f"({len(p_grid)} p-values x {len(T_grid)} T) ...")
        th = thermal_ratio_curve(sigma, T_grid, n_p, n_B)
        enh_grid = enh_vs_p(sigma, T_grid, p_grid, n_p, n_B)   # (T, p)

        # For every cell: interpolate enh(T;p) at p=cell, then peak over T of th*enh
        peak = np.zeros_like(P)
        peakT = np.zeros_like(P)
        for a in range(P.shape[0]):
            for b in range(P.shape[1]):
                pc = P[a, b]
                enh_T = np.array([np.interp(pc, p_grid, enh_grid[i, :])
                                  for i in range(len(T_grid))])
                ratio_T = th * enh_T
                k = int(np.argmax(ratio_T))
                peak[a, b] = ratio_T[k]
                peakT[a, b] = T_grid[k]

        # ignition threshold p* (1D, monotonic in p): peak1d(p) vs p
        peak1d = np.array([np.max(th * np.array(
            [np.interp(pp, p_grid, enh_grid[i, :]) for i in range(len(T_grid))]))
            for pp in p_grid])
        if np.any(peak1d >= 1.0) and np.any(peak1d < 1.0):
            p_star = float(np.interp(1.0, peak1d, p_grid))
        elif np.all(peak1d >= 1.0):
            p_star = 0.0          # ignites for all p in box
        else:
            p_star = np.inf       # never ignites in box

        # Putvinski-only reference: Coulomb alpha coupling with NO R-matrix
        # nuclear enhancement (include_belloni=False), at the nominal alpha
        # density. This is the most defensible "remove Belloni" benchmark and
        # is NOT a point on the multiplicative F_scale axis (different v-shape).
        cs.CROSS_SECTION = sigma
        enh_pu = np.array([
            mv.fp_coupled_calculation(
                T, pb.find_self_consistent_Te(n_p, n_B, T), n_p, n_B,
                include_alpha=True, n_alpha_over_ne=NOMINAL[0],
                include_belloni=False)['kinetic_enhancement']
            for T in T_grid])
        putv_peak = float(np.max(th * enh_pu))

        results[sigma] = dict(
            th=th, enh_grid=enh_grid, peak=peak, peakT=peakT,
            peak1d=peak1d, p_star=p_star, putv_peak=putv_peak,
        )

    return dict(n_p=n_p, n_B=n_B, T_grid=T_grid, na_axis=na_axis,
                fs_axis=fs_axis, NA=NA, FS=FS, P=P, p_grid=p_grid,
                results=results)


# ------------------------------------------------------------------
# Output: figure, CSV, summary
# ------------------------------------------------------------------
NOMINAL = (0.05, 1.0)        # code's operating point
CONSERVATIVE = (0.01, 0.5)   # defensible conservative point


def make_figure(M, args, out_png):
    na, fs = M['na_axis'], M['fs_axis']
    res = M['results']
    sigmas = args.cross_sections
    levels = np.linspace(0.6, 1.6, 21)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))

    # --- Row 0: filled P_F/P_B contour for each cross section ---
    for col, sigma in enumerate(sigmas[:2]):
        ax = axes[0, col]
        peak = res[sigma]['peak']
        cf = ax.contourf(na, fs, peak, levels=levels, cmap='RdYlGn', extend='both')
        cl = ax.contour(na, fs, peak, levels=[1.0], colors='k', linewidths=3)
        ax.clabel(cl, fmt='P_F/P_B=1', fontsize=9)
        # sub-ignition hatch
        ax.contourf(na, fs, peak, levels=[0, 1.0], colors='none',
                    hatches=['xxx'], alpha=0)
        ax.plot(*NOMINAL, marker='*', ms=20, mfc='white', mec='k', mew=1.6,
                ls='', label='code nominal (0.05, 1.0)')
        ax.plot(*CONSERVATIVE, marker='o', ms=11, mfc='cyan', mec='k', mew=1.5,
                ls='', label='conservative (0.01, 0.5)')
        ax.set_xlabel(r'$n_\alpha/n_e$', fontsize=12)
        ax.set_ylabel(r'Belloni $F$-factor scale', fontsize=12)
        ax.set_title(f'({chr(97+col)}) peak $P_F/P_B$ — {sigma} cross section',
                     fontsize=12)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.colorbar(cf, ax=ax, label='peak $P_F/P_B$')

    # --- Row 1, col 0: ignition-threshold overlay (both cross sections) ---
    ax = axes[1, 0]
    colors = {'TB': 'tab:blue', 'wang': 'tab:red'}
    for sigma in sigmas:
        peak = res[sigma]['peak']
        c = colors.get(sigma, 'k')
        ax.contour(na, fs, peak, levels=[1.0], colors=[c], linewidths=2.5)
        # shade sub-ignition (peak<1) for this sigma
        ax.contourf(na, fs, (peak < 1.0).astype(float), levels=[0.5, 1.5],
                    colors=[c], alpha=0.18)
    ax.plot(*NOMINAL, marker='*', ms=20, mfc='white', mec='k', mew=1.6, ls='')
    ax.plot(*CONSERVATIVE, marker='o', ms=11, mfc='cyan', mec='k', mew=1.5, ls='')
    ax.set_xlabel(r'$n_\alpha/n_e$', fontsize=12)
    ax.set_ylabel(r'Belloni $F$-factor scale', fontsize=12)
    ax.set_title('(c) ignition boundary ($P_F/P_B=1$) and sub-ignition region',
                 fontsize=12)
    legend_el = [Line2D([0], [0], color=colors.get(s, 'k'), lw=2.5,
                        label=f'{s}: $P_F/P_B=1$') for s in sigmas]
    legend_el += [
        Line2D([0], [0], marker='*', mfc='white', mec='k', ls='', ms=14,
               label='code nominal'),
        Line2D([0], [0], marker='o', mfc='cyan', mec='k', ls='', ms=9,
               label='conservative'),
    ]
    ax.legend(handles=legend_el, loc='upper right', fontsize=8, framealpha=0.9)

    # --- Row 1, col 1: 1D collapse (peak P_F/P_B vs p) ---
    ax = axes[1, 1]
    for sigma in sigmas:
        ax.semilogx(M['p_grid'][1:], res[sigma]['peak1d'][1:],
                    lw=2.5, color=colors.get(sigma, 'k'),
                    label=f'{sigma}  ($p^*$={res[sigma]["p_star"]:.3f})')
    ax.axhline(1.0, color='k', ls=':', lw=1.5, label='ignition threshold')
    ax.axvline(NOMINAL[0] * NOMINAL[1], color='gray', ls='--', lw=1.2,
               label=f'nominal p={NOMINAL[0]*NOMINAL[1]:.3f}')
    ax.axvline(CONSERVATIVE[0] * CONSERVATIVE[1], color='cyan', ls='--', lw=1.2,
               label=f'conservative p={CONSERVATIVE[0]*CONSERVATIVE[1]:.3f}')
    # Putvinski-only reference (Coulomb alpha, no R-matrix) at nominal n_alpha
    for sigma in sigmas:
        ax.axhline(res[sigma]['putv_peak'], color=colors.get(sigma, 'k'),
                   ls='-.', lw=1.2, alpha=0.7)
    pu_txt = ("Putvinski-only (Coulomb $\\alpha$, no R-matrix) @ "
              "$n_\\alpha/n_e$=0.05:\n"
              + " ;  ".join(f"{s} = {res[s]['putv_peak']:.2f}"
                            + ("" if res[s]['putv_peak'] >= 1 else " (sub-ign.)")
                            for s in sigmas))
    ax.text(0.03, 0.97, pu_txt, transform=ax.transAxes, fontsize=7.5,
            va='top', ha='left',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.85))
    ax.set_xlabel(r'$p = (n_\alpha/n_e)\times F_{\rm scale}$', fontsize=12)
    ax.set_ylabel('peak $P_F/P_B$', fontsize=12)
    ax.set_title('(d) degeneracy: peak $P_F/P_B$ is a function of $p$ alone',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('p-$^{11}$B kinetic ignition: sensitivity to alpha density and '
                 'R-matrix scattering\n'
                 f'($n_i$={args.n_i:.0e} cm$^{{-3}}$, $f_B$={args.f_B}, '
                 'kinetic-peak over $T_i$)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def write_csvs(M, args, grid_csv, summary_csv):
    res = M['results']
    with open(grid_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'n_alpha_over_ne', 'F_scale', 'p',
                    'peak_PF_PB', 'peak_Ti_keV', 'ignition'])
        for sigma in args.cross_sections:
            peak = res[sigma]['peak']
            peakT = res[sigma]['peakT']
            for a, fsv in enumerate(M['fs_axis']):
                for b, nav in enumerate(M['na_axis']):
                    w.writerow([sigma, f'{nav:.5f}', f'{fsv:.4f}',
                                f'{nav*fsv:.5f}', f'{peak[a, b]:.5f}',
                                f'{peakT[a, b]:.1f}', int(peak[a, b] > 1.0)])

    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'p_star_ignition_threshold',
                    'frac_box_sub_ignition', 'peak_at_nominal(0.05,1.0)',
                    'peak_at_conservative(0.01,0.5)', 'thermal_only_peak',
                    'putvinski_only_peak(no_Rmatrix,n_a=0.05)'])
        for sigma in args.cross_sections:
            peak = res[sigma]['peak']
            frac_sub = float(np.mean(peak <= 1.0))
            pk_nom = float(np.interp(NOMINAL[0] * NOMINAL[1], M['p_grid'],
                                     res[sigma]['peak1d']))
            pk_con = float(np.interp(CONSERVATIVE[0] * CONSERVATIVE[1],
                                     M['p_grid'], res[sigma]['peak1d']))
            th_only = float(np.max(res[sigma]['th']))
            w.writerow([sigma, f"{res[sigma]['p_star']:.5f}", f'{frac_sub:.4f}',
                        f'{pk_nom:.4f}', f'{pk_con:.4f}', f'{th_only:.4f}',
                        f"{res[sigma]['putv_peak']:.4f}"])


def print_summary(M, args):
    res = M['results']
    print("\n" + "=" * 74)
    print("SUMMARY — kinetic-peak P_F/P_B sensitivity")
    print("=" * 74)
    hdr = f"{'sigma':<6}{'p*':>9}{'%sub-ign':>10}{'nominal':>10}" \
          f"{'conserv':>10}{'thermal':>9}{'Putv-only':>11}"
    print(hdr)
    print("-" * 74)
    for sigma in args.cross_sections:
        peak = res[sigma]['peak']
        frac = 100.0 * float(np.mean(peak <= 1.0))
        pk_nom = float(np.interp(NOMINAL[0] * NOMINAL[1], M['p_grid'],
                                 res[sigma]['peak1d']))
        pk_con = float(np.interp(CONSERVATIVE[0] * CONSERVATIVE[1],
                                 M['p_grid'], res[sigma]['peak1d']))
        th = float(np.max(res[sigma]['th']))
        ps = res[sigma]['p_star']
        ps_s = 'all-ign' if ps == 0 else ('never' if not np.isfinite(ps)
                                          else f'{ps:.3f}')
        print(f"{sigma:<6}{ps_s:>9}{frac:>9.1f}%{pk_nom:>10.3f}"
              f"{pk_con:>10.3f}{th:>9.3f}{res[sigma]['putv_peak']:>11.3f}")
    print("-" * 74)
    print("nominal=(n_a/n_e=0.05, F=1.0); conserv=(0.01,0.5); "
          "Putv-only=Coulomb a, no R-matrix @ n_a=0.05")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cross-sections', nargs='+', default=['TB', 'wang'],
                    choices=['TB', 'wang'])
    ap.add_argument('--na-min', type=float, default=0.01)
    ap.add_argument('--na-max', type=float, default=0.10)
    ap.add_argument('--na-points', type=int, default=30)
    ap.add_argument('--fs-min', type=float, default=0.5)
    ap.add_argument('--fs-max', type=float, default=2.0)
    ap.add_argument('--fs-points', type=int, default=30)
    ap.add_argument('--p-points', type=int, default=24)
    ap.add_argument('--t-min', type=float, default=140.0)
    ap.add_argument('--t-max', type=float, default=420.0)
    ap.add_argument('--t-points', type=int, default=8)
    ap.add_argument('--n-i', type=float, default=1e14)
    ap.add_argument('--f-B', type=float, default=0.15)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(_THIS, '..', 'figures')))
    ap.add_argument('--verify-collapse', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_p = (1.0 - args.f_B) * args.n_i
    n_B = args.f_B * args.n_i

    print("=" * 74)
    print("p-11B kinetic-ignition 2D uncertainty propagation")
    print(f"  n_i={args.n_i:.0e}, f_B={args.f_B}; cross sections: {args.cross_sections}")
    print(f"  n_alpha/n_e in [{args.na_min}, {args.na_max}] ({args.na_points} pts)")
    print(f"  F-scale     in [{args.fs_min}, {args.fs_max}] ({args.fs_points} pts)")
    print("=" * 74)

    if args.verify_collapse:
        verify_collapse(n_p, n_B)

    M = build_maps(args)

    out_png = os.path.join(args.out, 'sensitivity_2d.png')
    grid_csv = os.path.join(args.out, 'sensitivity_2d_grid.csv')
    summary_csv = os.path.join(args.out, 'sensitivity_2d_summary.csv')
    make_figure(M, args, out_png)
    write_csvs(M, args, grid_csv, summary_csv)
    print_summary(M, args)
    print(f"\n[OK] figure : {out_png}")
    print(f"[OK] grid   : {grid_csv}")
    print(f"[OK] summary: {summary_csv}")


if __name__ == '__main__':
    main()
