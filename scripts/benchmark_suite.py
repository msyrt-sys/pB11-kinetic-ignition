#!/usr/bin/env python3
"""
benchmark_suite.py — numerical-foundations benchmark suite for the
                     "faithful-cross-section" revision (evidence-map document).

SESSION-2 deliverable. Independent of the Te/Ti decoupling scan
(te_ti_decoupling_scan.py); writes its own CSVs under figures/bench_*.csv so
there is no collision with session-1 outputs. Imports the UNMODIFIED physics
modules in src/ — no on-disk change to any module.

Tasks
-----
  1. CROSS-SECTION REACTIVITY BENCHMARK
     <sigma v>(T) in cm^3/s for Tentori-Belloni 2023 (TB), Wang 2026 (wang),
     and Nevins-Swain 2000 (NS) over T = 50..600 keV (50 keV step), all on the
     IDENTICAL Maxwell-Boltzmann integrator, so the column-to-column spread is
     a pure cross-section effect. Mutual ratios included. Putvinski 2019 is
     reported as a reference annotation only (no closed-form fit published;
     ~+20% over NS near 300 keV, ~= Wang).

  2. BREMSSTRAHLUNG-LIMITED THERMAL FLOOR
     Pure thermal Maxwellian P_F/P_B (R-matrix + suprathermal enhancement OFF),
     self-consistent T_e, f_B=0.15, Ti in [100,600] keV, for TB and Wang.
     Reports the maximum P_F/P_B and the Ti at which it occurs.

  3. THERMAL OPTIMUM SENSITIVITY
     2D thermal P_F/P_B over (Ti, f_B), f_B in [0.05,0.30], for TB and Wang.
     Locates the optimum (Ti*, f_B*) and the peak ratio per cross section.

  4. DEGENERACY RE-VERIFICATION + p* PINNING
     Independent-grid confirmation that the kinetic enhancement depends ONLY on
     p = (n_alpha/n_e) x F_scale (spread ~ 0), for both wang and TB, and a
     refined ignition-threshold p* (kinetic-peak P_F/P_B = 1) for each.

The NS cross section is reconstructed from the Nevins-Swain column of
Tentori & Belloni (2023) Nucl. Fusion 63 086001, Table 1 (verbatim), using the
SAME piecewise S-factor -> sigma form as the repo's sigma_TB. A continuity
self-check at the segment boundaries guards the reconstruction.

Usage
-----
  py -3.11 benchmark_suite.py --task all
  py -3.11 benchmark_suite.py --task 1
  py -3.11 benchmark_suite.py --task 4 --t-points 13 --p-points 20
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

# ------------------------------------------------------------------
# Imports (silence the import-time self-test prints of belloni / phaseshift)
# ------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_THIS, '..', 'src'))
sys.path.insert(0, _SRC)
sys.path.insert(0, _THIS)

_real_stdout = sys.stdout
sys.stdout = io.StringIO()
import cross_sections as cs            # noqa: E402
import power_balance as pb             # noqa: E402
import alpha_source as als            # noqa: E402
import main_validation as mv          # noqa: E402
import belloni_full_implementation    # noqa: E402,F401
import alpha_p_phaseshift             # noqa: E402,F401
sys.stdout = _real_stdout

# Shared physical constants pulled straight from the repo, so every column uses
# IDENTICAL constants (Gamow energy, reduced mass, unit conversions).
from cross_sections import (           # noqa: E402
    E_G_keV, barn_cm2, keV_to_erg, mu_pB11_g, sigma_TB, sigma_wang,
)

DEFAULT_NI = 1.0e14
DEFAULT_FB = 0.15


# ==================================================================
# Nevins-Swain (2000) cross section  -- reconstructed from TB 2023 Table 1
# (Nevins-Swain column, verbatim). Same S-factor -> sigma machinery as sigma_TB.
# Coefficients in MeV*b (as printed); energies in keV.
# ==================================================================
NS_C0, NS_C1, NS_C2 = 197.0, 0.240, 2.31e-4          # S1 polynomial
NS_AL, NS_EL, NS_dEL = 1.82e4, 148.0, 2.35           # S1 148 keV resonance
NS_D0, NS_D1, NS_D2, NS_D5 = 330.0, 66.1, -20.3, -1.58   # S2 polynomial
NS_B = 4.38                                          # S3 background
NS_A  = np.array([2.57e6, 5.67e5, 1.34e5, 5.68e5])   # S3 amplitudes (MeV*b)
NS_E  = np.array([581.3, 1083.0, 2405.0, 3344.0])    # S3 positions  (keV)
NS_dE = np.array([85.7, 234.0, 138.0, 309.0])        # S3 widths     (keV)
NS_E1, NS_E2, NS_E3 = 400.0, 642.0, 3500.0           # segment boundaries (keV)


def S_NS(E_keV):
    """Nevins-Swain piecewise astrophysical S-factor (MeV*b)."""
    E = np.atleast_1d(E_keV).astype(float)
    S = np.zeros_like(E)
    m1 = (E > 0) & (E <= NS_E1)
    m2 = (E > NS_E1) & (E <= NS_E2)
    m3 = (E > NS_E2) & (E <= NS_E3)
    S[m1] = (NS_C0 + NS_C1 * E[m1] + NS_C2 * E[m1]**2
             + NS_AL / ((E[m1] - NS_EL)**2 + NS_dEL**2))
    x = (E[m2] - NS_E1) / 100.0
    S[m2] = NS_D0 + NS_D1 * x + NS_D2 * x**2 + NS_D5 * x**5
    s = np.full_like(E[m3], NS_B)
    for k in range(4):
        s += NS_A[k] / ((E[m3] - NS_E[k])**2 + NS_dE[k]**2)
    S[m3] = s
    return S


def sigma_NS(E_cm_keV):
    """Nevins-Swain 2000 p-11B fusion cross section (barns). E_cm in keV (CM).

    sigma[b] = S[MeV*b] / E[MeV] * exp(-sqrt(E_G/E)), identical to sigma_TB.
    """
    E = np.atleast_1d(E_cm_keV).astype(float)
    sigma = np.zeros_like(E)
    mask = E > 0.5
    if np.any(mask):
        sigma[mask] = (S_NS(E[mask]) / (E[mask] / 1000.0)) * \
            np.exp(-np.sqrt(E_G_keV / E[mask]))
    return sigma


def _check_NS_continuity():
    """Guard: NS S-factor must be (near-)continuous at 400 and 642 keV.

    The genuine physical seam jumps are ~0.07%; the 0.5% tolerance passes those
    comfortably while still catching a coefficient-transcription regression.
    """
    for Eb in (NS_E1, NS_E2):
        lo = S_NS(Eb - 1e-3)[0]
        hi = S_NS(Eb + 1e-3)[0]
        rel = abs(hi - lo) / abs(lo)
        assert rel < 0.005, (f"NS S-factor discontinuity at {Eb} keV: "
                             f"{lo:.3f} vs {hi:.3f} ({rel:.2%})")
    return True


# ==================================================================
# General Maxwell-Boltzmann reactivity integrator
# EXACT replica of cross_sections.sigma_v_TB_numerical, but parameterized by an
# arbitrary sigma(E) function so TB/Wang/NS share identical integration.
# ==================================================================
def mb_reactivity(T_keV, sigma_func, E_max_keV=9760.0, n_points=10000):
    T_arr = np.atleast_1d(T_keV).astype(float)
    result = np.zeros_like(T_arr)
    for i, T in enumerate(T_arr):
        if T <= 0:
            continue
        E_low = np.logspace(np.log10(0.5), np.log10(50), 500)
        E_mid = np.linspace(50, 2000, n_points // 2)
        E_high = np.linspace(2000, E_max_keV, n_points // 2)
        E_grid = np.unique(np.concatenate([E_low, E_mid, E_high]))

        sig_cm2 = sigma_func(E_grid) * barn_cm2
        E_erg = E_grid * keV_to_erg
        boltz = np.exp(-E_grid / T)
        integrand = sig_cm2 * E_erg * boltz
        integral_val = np.trapezoid(integrand, E_erg)

        T_erg = T * keV_to_erg
        prefactor = np.sqrt(8.0 / (np.pi * mu_pB11_g)) / T_erg**1.5
        result[i] = prefactor * integral_val
    return result


# ==================================================================
# Belloni F-scale runtime injection (no on-disk edit), identical mechanism to
# sensitivity_analysis_2d.py. Used only by task 4.
# ==================================================================
_orig_belloni = als.belloni_2021_factor
_F_SCALE = [1.0]


def _scaled_belloni(v_proton_cm_s, T_p_keV):
    return _F_SCALE[0] * _orig_belloni(v_proton_cm_s, T_p_keV)


als.belloni_2021_factor = _scaled_belloni


def kinetic_enh(T_i, n_p, n_B, n_alpha_over_ne, f_scale=1.0, T_e=None,
                belloni=True):
    """enh = P_F[f_FP]/P_F[Maxwell] at one operating point (self-consistent Te
    unless T_e supplied)."""
    _F_SCALE[0] = f_scale
    if T_e is None:
        T_e = pb.find_self_consistent_Te(n_p, n_B, T_i)
    r = mv.fp_coupled_calculation(T_i, T_e, n_p, n_B, include_alpha=True,
                                  n_alpha_over_ne=n_alpha_over_ne,
                                  include_belloni=belloni)
    return r['kinetic_enhancement']


def _writerows(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[OK] {path}")


# ==================================================================
# TASK 1 — reactivity benchmark
# ==================================================================
def task1_reactivity(out_dir):
    print("\n" + "=" * 74)
    print("TASK 1 — CROSS-SECTION REACTIVITY BENCHMARK (<sigma v>, cm^3/s)")
    print("=" * 74)
    _check_NS_continuity()

    # Cross-check: mb_reactivity reproduces the repo integrator for TB and Wang.
    for name, sfun, glob in (("TB", sigma_TB, "TB"), ("wang", sigma_wang, "wang")):
        cs.CROSS_SECTION = glob
        a = mb_reactivity(300.0, sfun)[0]
        b = cs.sigma_v_TB_numerical(300.0)[0]
        rel = abs(a - b) / b
        print(f"  integrator self-check {name}(300 keV): mb={a:.4e} "
              f"repo={b:.4e}  rel={rel:.2e}  {'[OK]' if rel < 1e-9 else '[WARN]'}")

    T = np.arange(50, 601, 50, dtype=float)
    sv_TB = mb_reactivity(T, sigma_TB)
    sv_W = mb_reactivity(T, sigma_wang)
    sv_NS = mb_reactivity(T, sigma_NS)

    rows = []
    hdr = ['T_keV', 'sv_TB', 'sv_Wang', 'sv_NS',
           'ratio_TB_Wang', 'ratio_TB_NS', 'ratio_Wang_NS']
    for i, Ti in enumerate(T):
        rows.append([f'{Ti:.0f}', f'{sv_TB[i]:.4e}', f'{sv_W[i]:.4e}',
                     f'{sv_NS[i]:.4e}', f'{sv_TB[i]/sv_W[i]:.4f}',
                     f'{sv_TB[i]/sv_NS[i]:.4f}', f'{sv_W[i]/sv_NS[i]:.4f}'])
    _writerows(os.path.join(out_dir, 'bench_reactivity.csv'), hdr, rows)

    print(f"\n{'T':>5} | {'TB':>11} | {'Wang':>11} | {'NS':>11} | "
          f"{'TB/Wang':>8} | {'TB/NS':>7} | {'W/NS':>6}")
    print("-" * 74)
    for i, Ti in enumerate(T):
        print(f"{Ti:>5.0f} | {sv_TB[i]:>11.3e} | {sv_W[i]:>11.3e} | "
              f"{sv_NS[i]:>11.3e} | {sv_TB[i]/sv_W[i]:>8.3f} | "
              f"{sv_TB[i]/sv_NS[i]:>7.3f} | {sv_W[i]/sv_NS[i]:>6.3f}")
    print("\nPutvinski 2019: no published closed-form fit reproduced here; their "
          "kinetic-tail\nreassessment gives ~+20% over NS near 300 keV "
          "(~= Wang). Reference only.")
    return T, sv_TB, sv_W, sv_NS


# ==================================================================
# TASK 2 — bremsstrahlung-limited thermal floor
# ==================================================================
def task2_thermal_floor(out_dir, n_i, f_B, t_min=100.0, t_max=600.0, t_step=10.0):
    print("\n" + "=" * 74)
    print("TASK 2 — BREMSSTRAHLUNG-LIMITED THERMAL FLOOR (no FP, no R-matrix)")
    print(f"         f_B={f_B}, n_i={n_i:.0e}, self-consistent T_e")
    print("=" * 74)
    n_p = (1.0 - f_B) * n_i
    n_B = f_B * n_i
    T = np.arange(t_min, t_max + 0.5 * t_step, t_step, dtype=float)

    rows = []
    hdr = ['cross_section', 'Ti_keV', 'Te_keV', 'P_F_W_cm3', 'P_B_W_cm3',
           'PF_over_PB']
    peaks = {}
    for xs in ('TB', 'wang'):
        cs.CROSS_SECTION = xs
        ratios = np.zeros_like(T)
        for i, Ti in enumerate(T):
            info = pb.ignition_check(n_p, n_B, Ti)
            ratios[i] = info['P_F_over_P_Brem']
            rows.append([xs, f'{Ti:.0f}', f"{info['T_e_keV']:.2f}",
                         f"{info['P_F']:.4e}", f"{info['P_Brem']:.4e}",
                         f"{info['P_F_over_P_Brem']:.5f}"])
        k = int(np.argmax(ratios))
        peaks[xs] = (T[k], ratios[k])
    _writerows(os.path.join(out_dir, 'bench_thermal_floor.csv'), hdr, rows)

    print(f"\n{'cross section':>13} | {'max P_F/P_B':>12} | {'at Ti (keV)':>11} | "
          f"{'ignites?':>9}")
    print("-" * 56)
    for xs in ('TB', 'wang'):
        Tp, rp = peaks[xs]
        print(f"{xs:>13} | {rp:>12.4f} | {Tp:>11.0f} | "
              f"{'YES' if rp > 1 else 'no':>9}")
    return peaks


# ==================================================================
# TASK 3 — thermal optimum (Ti, f_B)
# ==================================================================
def task3_thermal_optimum(out_dir, n_i, t_min=150.0, t_max=600.0, t_points=19,
                          fb_min=0.05, fb_max=0.30, fb_points=21):
    print("\n" + "=" * 74)
    print("TASK 3 — THERMAL OPTIMUM (Ti, f_B), self-consistent T_e")
    print("=" * 74)
    T = np.linspace(t_min, t_max, t_points)
    FB = np.linspace(fb_min, fb_max, fb_points)

    grid_rows = []
    grid_hdr = ['cross_section', 'f_B', 'Ti_keV', 'PF_over_PB']
    summary = {}
    for xs in ('TB', 'wang'):
        cs.CROSS_SECTION = xs
        R = np.zeros((len(FB), len(T)))
        for j, fb in enumerate(FB):
            n_p = (1.0 - fb) * n_i
            n_B = fb * n_i
            for i, Ti in enumerate(T):
                r = pb.ignition_check(n_p, n_B, Ti)['P_F_over_P_Brem']
                R[j, i] = r
                grid_rows.append([xs, f'{fb:.4f}', f'{Ti:.1f}', f'{r:.5f}'])
        j_opt, i_opt = np.unravel_index(np.argmax(R), R.shape)
        summary[xs] = dict(fB=FB[j_opt], Ti=T[i_opt], peak=R[j_opt, i_opt])
    _writerows(os.path.join(out_dir, 'bench_thermal_optimum_grid.csv'),
               grid_hdr, grid_rows)

    srows = []
    shdr = ['cross_section', 'opt_f_B', 'opt_Ti_keV', 'peak_PF_over_PB',
            'ignites']
    print(f"\n{'cross section':>13} | {'opt f_B':>8} | {'opt Ti':>7} | "
          f"{'peak P_F/P_B':>12} | {'ignites':>8}")
    print("-" * 62)
    for xs in ('TB', 'wang'):
        s = summary[xs]
        srows.append([xs, f"{s['fB']:.4f}", f"{s['Ti']:.1f}",
                      f"{s['peak']:.5f}", int(s['peak'] > 1)])
        print(f"{xs:>13} | {s['fB']:>8.3f} | {s['Ti']:>7.0f} | "
              f"{s['peak']:>12.4f} | {'YES' if s['peak'] > 1 else 'no':>8}")
    _writerows(os.path.join(out_dir, 'bench_thermal_optimum_summary.csv'),
               shdr, srows)
    return summary


# ==================================================================
# TASK 4 — degeneracy re-verification + p* pinning
# ==================================================================
def task4_degeneracy_pstar(out_dir, n_i, f_B, t_min=140.0, t_max=440.0,
                           t_points=13, p_points=18, p_max=0.10):
    print("\n" + "=" * 74)
    print("TASK 4 — DEGENERACY (enh = f(p) only) + IGNITION THRESHOLD p*")
    print("=" * 74)
    n_p = (1.0 - f_B) * n_i
    n_B = f_B * n_i

    # ----- (a) independent-grid degeneracy check (fresh equal-p tuples) -----
    # Different from sensitivity_analysis_2d.verify_collapse (which used
    # p=0.05 and p=0.02). Here: p in {0.03, 0.04, 0.06} via off-diagonal pairs.
    T_probe = 200.0
    groups = {
        0.03: [(0.03, 1.0), (0.06, 0.5), (0.015, 2.0), (0.12, 0.25)],
        0.04: [(0.04, 1.0), (0.02, 2.0), (0.08, 0.5)],
        0.06: [(0.06, 1.0), (0.12, 0.5), (0.03, 2.0), (0.24, 0.25)],
    }
    chk_rows = []
    chk_hdr = ['cross_section', 'p', 'n_alpha_over_ne', 'F_scale', 'enh',
               'group_spread']
    print(f"  degeneracy check at Ti={T_probe:.0f} keV (self-consistent Te):")
    deg_ok = True
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        Te = pb.find_self_consistent_Te(n_p, n_B, T_probe)
        print(f"  [{xs}]  Te_sc={Te:.1f} keV")
        for p, pairs in groups.items():
            vals = [kinetic_enh(T_probe, n_p, n_B, na, fs, T_e=Te)
                    for (na, fs) in pairs]
            spread = max(vals) - min(vals)
            deg_ok = deg_ok and spread < 1e-6
            for (na, fs), v in zip(pairs, vals):
                chk_rows.append([xs, f'{p:.4f}', f'{na:.4f}', f'{fs:.3f}',
                                 f'{v:.8f}', f'{spread:.2e}'])
            tag = 'PASS' if spread < 1e-6 else 'FAIL'
            print(f"     p={p:.3f}: " + ", ".join(f"{v:.5f}" for v in vals)
                  + f"   spread={spread:.1e}  [{tag}]")
    _writerows(os.path.join(out_dir, 'bench_degeneracy_check.csv'),
               chk_hdr, chk_rows)
    print(f"  => degeneracy {'CONFIRMED (spread~0)' if deg_ok else 'FAILED'}")

    # ----- (b) p* : kinetic-peak P_F/P_B(p) = 1 for each cross section -----
    T = np.linspace(t_min, t_max, t_points)
    p_floor = p_max * 1e-3
    p_grid = np.unique(np.concatenate([[0.0],
                                       np.geomspace(p_floor, p_max, p_points)]))
    star_rows = []
    star_hdr = ['cross_section', 'p', 'kinetic_peak_PF_PB', 'peak_Ti_keV']
    pstars = {}
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        Te_sc = np.array([pb.find_self_consistent_Te(n_p, n_B, Ti) for Ti in T])
        th = np.array([pb.ignition_check(n_p, n_B, Ti,
                                         T_e_keV=Te_sc[i])['P_F_over_P_Brem']
                       for i, Ti in enumerate(T)])
        # enh(Ti, p) with p == n_alpha_over_ne, F_scale=1 (degeneracy exploited)
        enh = np.zeros((len(T), len(p_grid)))
        for j, p in enumerate(p_grid):
            for i, Ti in enumerate(T):
                enh[i, j] = kinetic_enh(Ti, n_p, n_B, p, 1.0, T_e=Te_sc[i])
        peak1d = np.array([np.max(th * enh[:, j]) for j in range(len(p_grid))])
        peakTi = np.array([T[int(np.argmax(th * enh[:, j]))]
                           for j in range(len(p_grid))])
        for j, p in enumerate(p_grid):
            star_rows.append([xs, f'{p:.5f}', f'{peak1d[j]:.5f}',
                              f'{peakTi[j]:.1f}'])
        # p* : linear interpolation of peak1d(p) crossing 1.0 (monotone in p)
        if np.any(peak1d >= 1.0) and np.any(peak1d < 1.0):
            p_star = float(np.interp(1.0, peak1d, p_grid))
        elif np.all(peak1d >= 1.0):
            p_star = 0.0
        else:
            p_star = float('inf')
        pstars[xs] = dict(p_star=p_star,
                          peak_at_nominal=float(np.interp(0.05, p_grid, peak1d)),
                          thermal_floor=float(peak1d[0]))
    _writerows(os.path.join(out_dir, 'bench_pstar.csv'), star_hdr, star_rows)

    print(f"\n{'cross section':>13} | {'p*':>9} | {'peak@p=0.05':>11} | "
          f"{'p=0 floor':>9}")
    print("-" * 52)
    for xs in ('wang', 'TB'):
        s = pstars[xs]
        ps = ('all-ign' if s['p_star'] == 0 else
              ('never' if not np.isfinite(s['p_star']) else f"{s['p_star']:.4f}"))
        print(f"{xs:>13} | {ps:>9} | {s['peak_at_nominal']:>11.4f} | "
              f"{s['thermal_floor']:>9.4f}")
    return deg_ok, pstars


# ==================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', default='all',
                    choices=['all', '1', '2', '3', '4'])
    ap.add_argument('--n-i', type=float, default=DEFAULT_NI)
    ap.add_argument('--f-B', type=float, default=DEFAULT_FB)
    ap.add_argument('--t-points', type=int, default=13, help='task4 Ti points')
    ap.add_argument('--p-points', type=int, default=18, help='task4 p points')
    ap.add_argument('--out', default=os.path.normpath(
        os.path.join(_THIS, '..', 'figures')))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("#" * 74)
    print("p-11B faithful-cross-section numerical benchmark suite (session-2)")
    print(f"  n_i={args.n_i:.0e} cm^-3, f_B(default)={args.f_B}; out={args.out}")
    print("#" * 74)

    if args.task in ('all', '1'):
        task1_reactivity(args.out)
    if args.task in ('all', '2'):
        task2_thermal_floor(args.out, args.n_i, args.f_B)
    if args.task in ('all', '3'):
        task3_thermal_optimum(args.out, args.n_i)
    if args.task in ('all', '4'):
        task4_degeneracy_pstar(args.out, args.n_i, args.f_B,
                               t_points=args.t_points, p_points=args.p_points)

    print("\n" + "#" * 74)
    print("BENCHMARK SUITE COMPLETE")
    print("#" * 74)


if __name__ == '__main__':
    main()
