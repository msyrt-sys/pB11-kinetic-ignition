#!/usr/bin/env python3
"""
te_ti_decoupling_scan.py — two-lever (K1 x K2) scan for p-11B kinetic ignition.

Explores whether Te/Ti decoupling (K1) and the R-matrix kinetic enhancement (K2)
synergistically expand the ignition region with the Wang 2026 cross section.

  x-axis : p   = (n_alpha/n_e) x F_scale  in [0, 0.10]   (K2 strength; degenerate
                 product established by sensitivity_analysis_2d.py)
  y-axis : tau = Te / Ti                  in [0.30, 1.0] (K1 decoupling)

CRITICAL CONTEXT (verified, see header of the summary printout):
  The baseline model ALREADY solves a self-consistent Te (electron power balance
  P_alpha_e + P_ie = P_brems), giving tau_sc = Te/Ti ~ 0.34-0.53 (mean ~0.42).
  So K1 decoupling is already fully ON in every prior result; the 18.9% Wang
  sub-ignition map corresponds to tau ~ tau_sc, NOT tau = 1.0. tau = 1.0 (Te=Ti,
  no decoupling) is in fact catastrophic (thermal P_F/P_B ~ 0.37).

  This script overrides Te = tau x Ti to scan DEPARTURES from self-consistency.
  HONESTY CAVEAT: forcing tau < tau_sc means imposing electrons COLDER than the
  electron power balance allows. The required electron heat removal (transport /
  cooling) is NOT charged against P_F/P_brems, so tau < tau_sc is an
  optimistic, cost-unaccounted assumption. The self-consistent band is the
  physically honest operating region; it is shaded on the figure.

Outputs (to --out, default ../figures):
  - te_ti_decoupling.png            : (p x tau) map + 1D tau curves
  - te_ti_decoupling_grid.csv       : raw (p, tau) peak P_F/P_B grid
  - te_ti_decoupling_curves.csv     : 1D P_F/P_B vs tau for 3 scenarios

Usage:
  python te_ti_decoupling_scan.py
  python te_ti_decoupling_scan.py --tau-points 12 --p-points 13 --cross-section wang
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_THIS, '..', 'src'))
sys.path.insert(0, _SRC)
sys.path.insert(0, _THIS)

_real = sys.stdout
sys.stdout = io.StringIO()
import cross_sections as cs           # noqa: E402
import power_balance as pb            # noqa: E402
import alpha_source as als            # noqa: E402
import main_validation as mv          # noqa: E402
import belloni_full_implementation    # noqa: E402,F401
import alpha_p_phaseshift             # noqa: E402,F401
sys.stdout = _real

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402

NOMINAL_P = 0.05        # n_alpha=0.05, F=1.0
CONSERV_P = 0.005       # n_alpha=0.01, F=0.5


def thermal_ratio(n_p, n_B, Ti, tau):
    """Thermal P_F/P_B with Te = tau*Ti (override)."""
    return pb.ignition_check(n_p, n_B, Ti, T_e_keV=tau * Ti)['P_F_over_P_Brem']


def enh(n_p, n_B, Ti, tau, p, belloni=True):
    """Kinetic enhancement at Te = tau*Ti, alpha density p (F_scale=1)."""
    return mv.fp_coupled_calculation(
        Ti, tau * Ti, n_p, n_B, include_alpha=True, n_alpha_over_ne=p,
        include_belloni=belloni)['kinetic_enhancement']


def kinetic_peak_over_Ti(n_p, n_B, T_grid, tau, p, enh_col=None):
    """max_Ti thermal(Ti,tau)*enh(Ti,tau,p). enh_col may be precomputed."""
    best = -np.inf
    for i, Ti in enumerate(T_grid):
        e = enh_col[i] if enh_col is not None else enh(n_p, n_B, Ti, tau, p)
        best = max(best, thermal_ratio(n_p, n_B, Ti, tau) * e)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cross-section', default='wang', choices=['TB', 'wang'])
    ap.add_argument('--p-min', type=float, default=0.0)
    ap.add_argument('--p-max', type=float, default=0.10)
    ap.add_argument('--p-points', type=int, default=13)
    ap.add_argument('--tau-min', type=float, default=0.30)
    ap.add_argument('--tau-max', type=float, default=1.0)
    ap.add_argument('--tau-points', type=int, default=11)
    ap.add_argument('--t-min', type=float, default=160.0)
    ap.add_argument('--t-max', type=float, default=440.0)
    ap.add_argument('--t-points', type=int, default=7)
    ap.add_argument('--n-i', type=float, default=1e14)
    ap.add_argument('--f-B', type=float, default=0.15)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(_THIS, '..', 'figures')))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cs.CROSS_SECTION = args.cross_section
    n_i = args.n_i
    n_p = (1 - args.f_B) * n_i
    n_B = args.f_B * n_i

    T_grid = np.linspace(args.t_min, args.t_max, args.t_points)
    tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_points)
    p_floor = max(args.p_min, args.p_max * 1e-3)
    p_grid = np.unique(np.concatenate([[0.0], np.geomspace(p_floor, args.p_max,
                                                           args.p_points)]))

    # self-consistent tau_sc(Ti) for the physical band
    tau_sc = np.array([pb.find_self_consistent_Te(n_p, n_B, Ti) / Ti
                       for Ti in T_grid])

    print("=" * 72)
    print(f"Te/Ti decoupling (K1) x R-matrix enhancement (K2) — '{args.cross_section}'")
    print(f"  self-consistent tau_sc = Te/Ti over Ti: "
          f"{tau_sc.min():.3f}-{tau_sc.max():.3f} (mean {tau_sc.mean():.3f})")
    print(f"  -> baseline results already use tau ~ tau_sc (decoupling ON)")
    print("=" * 72)

    # enh(Ti, p) per tau (Te-dependent; p-collapse holds at fixed tau)
    peak = np.zeros((len(tau_grid), len(p_grid)))    # (tau, p)
    print(f"  computing enh on {len(tau_grid)} tau x {len(p_grid)} p x "
          f"{len(T_grid)} Ti ...")
    enh_by_tau = {}
    for a, tau in enumerate(tau_grid):
        enh_Tp = np.zeros((len(T_grid), len(p_grid)))
        for j, p in enumerate(p_grid):
            for i, Ti in enumerate(T_grid):
                enh_Tp[i, j] = enh(n_p, n_B, Ti, tau, p)
        enh_by_tau[a] = enh_Tp
        th = np.array([thermal_ratio(n_p, n_B, Ti, tau) for Ti in T_grid])
        for j in range(len(p_grid)):
            peak[a, j] = np.max(th * enh_Tp[:, j])

    # 1D curves vs tau: nominal p, conservative p (full K2), Putvinski-only (K2 off)
    curve_nom = np.zeros(len(tau_grid))
    curve_con = np.zeros(len(tau_grid))
    curve_pu = np.zeros(len(tau_grid))
    for a, tau in enumerate(tau_grid):
        th = np.array([thermal_ratio(n_p, n_B, Ti, tau) for Ti in T_grid])
        enh_Tp = enh_by_tau[a]
        curve_nom[a] = np.max(th * np.array(
            [np.interp(NOMINAL_P, p_grid, enh_Tp[i, :]) for i in range(len(T_grid))]))
        curve_con[a] = np.max(th * np.array(
            [np.interp(CONSERV_P, p_grid, enh_Tp[i, :]) for i in range(len(T_grid))]))
        curve_pu[a] = np.max([th[i] * enh(n_p, n_B, Ti, tau, NOMINAL_P,
                                          belloni=False)
                              for i, Ti in enumerate(T_grid)])

    # % of the standard (n_alpha,F) box sub-ignition vs tau
    na_box = np.linspace(0.01, 0.10, 25)
    fs_box = np.linspace(0.0, 2.0, 25)
    NAb, FSb = np.meshgrid(na_box, fs_box)
    Pbox = (NAb * FSb).ravel()
    frac_sub = np.array([
        100.0 * np.mean(np.interp(Pbox, p_grid, peak[a, :]) <= 1.0)
        for a in range(len(tau_grid))])

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6))
    band_lo, band_hi = tau_sc.min(), tau_sc.max()

    ax = axes[0]
    levels = np.linspace(0.4, 1.6, 25)
    cf = ax.contourf(p_grid, tau_grid, peak, levels=levels, cmap='RdYlGn',
                     extend='both')
    ax.contour(p_grid, tau_grid, peak, levels=[1.0], colors='k', linewidths=3)
    ax.axhspan(band_lo, band_hi, color='royalblue', alpha=0.18, zorder=1)
    ax.axhline(tau_sc.mean(), color='royalblue', ls='--', lw=1.6)
    ax.text(args.p_max * 0.98, tau_sc.mean(), ' self-consistent Te\n (physical)',
            color='royalblue', fontsize=8.5, va='center', ha='right')
    ax.plot(NOMINAL_P, tau_sc.mean(), marker='*', ms=18, mfc='white', mec='k',
            mew=1.6, ls='')
    ax.plot(CONSERV_P, tau_sc.mean(), marker='o', ms=10, mfc='cyan', mec='k',
            mew=1.4, ls='')
    ax.annotate('optimistic\n(forced cooling,\ncost not charged)',
                xy=(args.p_max * 0.5, args.tau_min + 0.01), fontsize=8,
                color='darkred', va='bottom', ha='center')
    ax.set_xlabel(r'$p = (n_\alpha/n_e)\times F_{\rm scale}$  (K2)', fontsize=12)
    ax.set_ylabel(r'$\tau = T_e/T_i$  (K1 decoupling)', fontsize=12)
    ax.set_title(f'(a) kinetic-peak $P_F/P_B$ — {args.cross_section}', fontsize=12)
    plt.colorbar(cf, ax=ax, label='peak $P_F/P_B$')

    ax = axes[1]
    ax.plot(tau_grid, curve_pu, 'o-', color='tab:gray', lw=2,
            label='K1 only (Putvinski $\\alpha$, K2 off, $n_\\alpha$=0.05)')
    ax.plot(tau_grid, curve_con, 's-', color='tab:orange', lw=2,
            label='K1 + K2 conservative ($p$=0.005)')
    ax.plot(tau_grid, curve_nom, '^-', color='tab:green', lw=2.2,
            label='K1 + K2 nominal ($p$=0.05)')
    ax.axhline(1.0, color='k', ls=':', lw=1.5)
    ax.axvspan(band_lo, band_hi, color='royalblue', alpha=0.18,
               label='self-consistent $\\tau$ (physical)')
    ax.set_xlabel(r'$\tau = T_e/T_i$', fontsize=12)
    ax.set_ylabel('kinetic-peak $P_F/P_B$', fontsize=12)
    ax.set_title('(b) ignition vs decoupling, by lever combination', fontsize=12)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # more decoupling (lower tau) to the right

    fig.suptitle('Two-lever synergy: $T_e/T_i$ decoupling (K1) x R-matrix '
                 f'enhancement (K2) — Wang cross section\n($n_i$={n_i:.0e}, '
                 f'$f_B$={args.f_B})', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = os.path.join(args.out, 'te_ti_decoupling.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---------------- CSV ----------------
    grid_csv = os.path.join(args.out, 'te_ti_decoupling_grid.csv')
    with open(grid_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tau', 'p', 'peak_PF_PB', 'ignition'])
        for a, tau in enumerate(tau_grid):
            for j, p in enumerate(p_grid):
                w.writerow([f'{tau:.4f}', f'{p:.5f}', f'{peak[a, j]:.5f}',
                            int(peak[a, j] > 1.0)])
    curves_csv = os.path.join(args.out, 'te_ti_decoupling_curves.csv')
    with open(curves_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tau', 'K1_only_Putvinski', 'K1K2_conservative_p0.005',
                    'K1K2_nominal_p0.05', 'pct_box_sub_ignition'])
        for a, tau in enumerate(tau_grid):
            w.writerow([f'{tau:.4f}', f'{curve_pu[a]:.4f}', f'{curve_con[a]:.4f}',
                        f'{curve_nom[a]:.4f}', f'{frac_sub[a]:.1f}'])

    # ---------------- summary ----------------
    def thr(curve):
        # curve decreases with tau; ignition holds for tau <= thr. Return the
        # interpolated crossing (largest tau with P_F/P_B >= 1).
        above = curve >= 1.0
        if not above.any():
            return None
        if above.all():
            return tau_grid[-1]
        idx = int(np.where(above)[0].max())
        if idx == len(tau_grid) - 1:
            return tau_grid[-1]
        t0, t1 = tau_grid[idx], tau_grid[idx + 1]
        c0, c1 = curve[idx], curve[idx + 1]
        return float(t0 + (1.0 - c0) * (t1 - t0) / (c1 - c0))

    print("\nSUMMARY (kinetic-peak P_F/P_B vs tau):")
    print(f"{'tau':>6}{'K1only(Putv)':>14}{'K1K2 consv':>12}"
          f"{'K1K2 nom':>11}{'%box sub-ign':>14}")
    for a, tau in enumerate(tau_grid):
        mark = ' <- self-consistent' if band_lo <= tau <= band_hi else ''
        print(f"{tau:>6.2f}{curve_pu[a]:>14.3f}{curve_con[a]:>12.3f}"
              f"{curve_nom[a]:>11.3f}{frac_sub[a]:>13.1f}%{mark}")
    print(f"\nself-consistent tau band: [{band_lo:.3f}, {band_hi:.3f}]")
    t_pu = thr(curve_pu)
    t_con = thr(curve_con)
    t_nom = thr(curve_nom)
    fmt = lambda t: f"tau <= {t:.3f}" if t is not None else "never (in range)"
    print(f"K1-alone (Putvinski, K2 off, n_a=0.05) ignites for {fmt(t_pu)}")
    print(f"K1+K2 conservative corner (p=0.005)    ignites for {fmt(t_con)}")
    print(f"K1+K2 nominal (p=0.05)                 ignites for {fmt(t_nom)}")
    print(f"(self-consistent band tau in [{band_lo:.3f}, {band_hi:.3f}])")
    print(f"\n[OK] {out_png}\n[OK] {grid_csv}\n[OK] {curves_csv}")


if __name__ == '__main__':
    main()
