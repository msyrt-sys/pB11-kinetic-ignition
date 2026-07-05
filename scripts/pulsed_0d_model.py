#!/usr/bin/env python3
"""
pulsed_0d_model.py — time-dependent (pulsed) two-temperature 0-D p-11B model.

Tests what the steady-state Monte Carlo could not: a TRANSIENT Te < Ti window.
Rider's net-negative result is a steady-state statement; a short pulse
(tau_pulse < tau_ie) keeps electrons cold (low bremsstrahlung) while ions are
hot (high fusion). Does the time-integrated energy gain exceed unity?

Two-temperature 0-D rate+energy ODEs (no spatial dependence), state vector
  y = [U_i, U_e, n_p, n_B, n_alpha, E_fus, E_brems, E_trans]   (erg/cm^3, cm^-3)
  U_i = 3/2 (n_p+n_B) Ti ;  U_e = 3/2 n_e Te ;  n_e = n_p + 5 n_B + 2 n_alpha
  dU_i/dt = (1-frac_e)*P_fus  - P_ie - U_i/tau_E - (fuel thermal removal)
  dU_e/dt = frac_e*P_fus      + P_ie - P_brems - U_e/tau_E
  dn_p/dt = -R ; dn_B/dt = -R ; dn_alpha/dt = 3R - n_alpha/tau_ash
  P_ie = Spitzer/Trubnikov e-i energy equilibration (drives Te -> Ti)
  P_fus = R*Q (8.68 MeV, all charged -> plasma)

Spark: instantaneous. At t=0 ions are at Ti_init, electrons at Te_init=ratio*Ti.
  E_spark = U_i(0) + U_e(0).

*** HONESTY (cycle-integrated, NOTHING dropped): ***
  G = E_fus / (E_spark + E_brems + E_trans)
  - E_spark counted (the pulsed analogue of P_drive).
  - FINITE tau_E: transport loss U/tau_E is ALWAYS on; no perfect confinement.
  - PERFECT-CONFINEMENT ARTIFACT: as tau_E -> inf, E_trans -> 0 and G inflates
    artificially. The realistic achievable tau_E for p-11B at ~1e14 cm^-3 is of
    order ~1 s (magnetic; Lawson for p-11B steady ignition needs ~10-100 s,
    which is NOT achievable). The G=1 contour's tau_E is compared to this band;
    if G>1 only requires tau_E >> achievable, the island is a confinement artifact.

CAVEATS (optimistic-biased -> real G LOWER):
  * E_spark = ideal thermal energy; real driver efficiency < 1.
  * Deep transient window (low Te_init) is costlier/harder to create; that setup
    cost is not modelled.
  * Prompt alpha deposition (slowing-down not resolved) upper-bounds ion heating.
  * Ash accumulates (tau_ash = inf) -> rising brems (the honest, harsher choice).
  * Channeling scenario redirects alpha power to ions with NO drive cost (most
    optimistic) and no tail term; if even that fails, the result is robust.

MANDATORY <sigma v>(Ti) / R(Te) interpolation self-check runs first (< 0.1%).

Outputs (to --out): pulsed_timeseries.png, pulsed_Gmap.png,
pulsed_Gmap.csv, pulsed_summary.csv
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
from collision_operators import m_p_g, m_B_g, m_e_g, e_esu, Z_B   # noqa: E402
from power_balance import E_rest_keV  # noqa: E402
sys.stdout = _real

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

keV_to_erg = cs.keV_to_erg
Q_erg = cs.Q_pB11_keV * keV_to_erg
LNL = 17.0
N_I = 1e14
TAU_E_ACHIEVABLE = 1.0      # s, generous magnetic-confinement reference at 1e14
TAU_E_BAND = (0.1, 1.0)     # s


def build_interpolators():
    Ti_grid = np.linspace(80.0, 850.0, 1541)     # 0.5 keV
    Te_grid = np.linspace(5.0, 850.0, 1691)       # 0.5 keV
    SV = {}
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        SV[xs] = cs.sigma_v_TB_numerical(Ti_grid)
    R_grid = np.array([pb.relativistic_R_factor(T) for T in Te_grid])
    print("=" * 72)
    print("INTERPOLATION SELF-CHECK (< 0.1% vs full integrator)")
    print("=" * 72)
    ok = True
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        for T in (213.7, 333.3, 457.1, 611.9, 744.4):
            full = cs.sigma_v_TB_numerical(T)[0]
            ip = float(np.interp(T, Ti_grid, SV[xs]))
            err = abs(ip - full) / full * 100
            ok &= err < 0.1
            print(f"  <sv> {xs:<4} Ti={T:6.1f}: err={err:.4f}%  "
                  f"{'OK' if err < 0.1 else 'FAIL'}")
    for T in (37.0, 123.0, 256.0, 444.0, 700.0):
        full = pb.relativistic_R_factor(T)
        ip = float(np.interp(T, Te_grid, R_grid))
        err = abs(ip - full) / full * 100
        ok &= err < 0.1
        print(f"  R(Te) Te={T:6.1f}: err={err:.4f}%  {'OK' if err < 0.1 else 'FAIL'}")
    if not ok:
        print("\nSELF-CHECK FAILED — aborting.")
        sys.exit(2)
    print("SELF-CHECK PASSED.\n")
    return Ti_grid, SV, Te_grid, R_grid


def make_rhs(sv_x, Ti_grid, R_grid, Te_grid, tau_E, tau_ash, eta_ch):
    cgi = 1.5 * keV_to_erg
    base_pref = (4 * np.sqrt(2 * np.pi) / 3) * e_esu**4 * LNL / (m_e_g**0.5)

    def rhs(t, y):
        U_i, U_e, n_p, n_B, n_a, _Ef, _Eb, _Et = y
        n_p = max(n_p, 1e6); n_B = max(n_B, 1e6); n_a = max(n_a, 0.0)
        n_e = n_p + Z_B * n_B + 2 * n_a
        cap_i = cgi * (n_p + n_B)
        cap_e = cgi * n_e
        Ti = max(U_i / cap_i, 0.05)
        Te = max(U_e / cap_e, 0.05)
        sv = np.interp(Ti, Ti_grid, sv_x)
        R = n_p * n_B * sv
        P_fus = R * Q_erg
        frac_e = min(0.15 * (150.0 / max(Te, 50.0))**0.5, 0.5)
        # channeling: move eta_ch of the electron-share to ions (no drive cost)
        P_ae = (1 - eta_ch) * frac_e * P_fus
        P_ai = P_fus - P_ae
        Rr = np.interp(Te, Te_grid, R_grid)
        base = base_pref * n_e / (Te * keV_to_erg)**1.5
        nu_pe = (m_e_g / m_p_g) * base
        nu_Be = (m_e_g / m_B_g) * base * Z_B**2
        P_ie = (1.5 * n_p * nu_pe + 1.5 * n_B * nu_Be) * (Ti - Te) * keV_to_erg * Rr
        Zeff = (n_p + n_B * Z_B**2 + n_a * 4) / n_e
        Pcl = 5.34e-31 * n_e**2 * Zeff * np.sqrt(Te)
        x = Te / E_rest_keV
        g = (1 + 1.78 * x**1.34) + 2.12 * x * (1 + 1.1 * x + x**2
                                               - 1.25 * x**2.5) / Zeff
        P_brems = Pcl * g * 1e7                      # W/cm3 -> erg/s/cm3
        Lt_i = U_i / tau_E
        Lt_e = U_e / tau_E
        fuel_rem = 3.0 * R * Ti * keV_to_erg          # consumed-fuel thermal energy
        dUi = P_ai - P_ie - Lt_i - fuel_rem
        dUe = P_ae + P_ie - P_brems - Lt_e
        dnp = -R
        dnB = -R
        dna = 3 * R - (n_a / tau_ash if np.isfinite(tau_ash) else 0.0)
        return [dUi, dUe, dnp, dnB, dna, P_fus, P_brems, Lt_i + Lt_e]
    return rhs


def run_pulse(sv_x, Ti_grid, R_grid, Te_grid, Ti_init, Te_ratio, tau_E,
              eta_ch=0.0, tau_ash=np.inf, f_B=0.15):
    n_p0 = (1 - f_B) * N_I
    n_B0 = f_B * N_I
    n_a0 = 0.0
    n_e0 = n_p0 + Z_B * n_B0
    Te_init = Te_ratio * Ti_init
    U_i0 = 1.5 * (n_p0 + n_B0) * Ti_init * keV_to_erg
    U_e0 = 1.5 * n_e0 * Te_init * keV_to_erg
    E_spark = U_i0 + U_e0
    # e-i equilibration time scale at the initial state
    base = (4 * np.sqrt(2 * np.pi) / 3) * e_esu**4 * LNL / (m_e_g**0.5) \
        * n_e0 / (Te_init * keV_to_erg)**1.5
    Rr = pb.relativistic_R_factor(Te_init)
    P_ie0 = (1.5 * n_p0 * (m_e_g / m_p_g) * base
             + 1.5 * n_B0 * (m_e_g / m_B_g) * base * Z_B**2) \
        * (Ti_init - Te_init) * keV_to_erg * Rr
    tau_ie = U_i0 / P_ie0 if P_ie0 > 0 else np.inf

    rhs = make_rhs(sv_x, Ti_grid, R_grid, Te_grid, tau_E, tau_ash, eta_ch)
    y0 = [U_i0, U_e0, n_p0, n_B0, n_a0, 0.0, 0.0, 0.0]
    t_max = min(60 * max(tau_E, tau_ie), 2000.0)

    def ev_cold(t, y):
        Ti = y[0] / (1.5 * keV_to_erg * (max(y[2], 1e6) + max(y[3], 1e6)))
        return Ti - 80.0
    ev_cold.terminal = True
    ev_cold.direction = -1

    sol = solve_ivp(rhs, (0, t_max), y0, method='Radau', events=ev_cold,
                    rtol=1e-6, atol=[1e0, 1e0, 1e6, 1e6, 1e6, 1e0, 1e0, 1e0],
                    dense_output=True, max_step=t_max / 20)
    E_fus = sol.y[5, -1]
    E_brems = sol.y[6, -1]
    E_trans = sol.y[7, -1]
    G = E_fus / (E_spark + E_brems + E_trans)

    # time series + transient diagnostics
    tt = np.linspace(0, sol.t[-1], 400)
    Y = sol.sol(tt)
    n_p, n_B, n_a = Y[2], Y[3], Y[4]
    n_e = n_p + Z_B * n_B + 2 * n_a
    Ti = Y[0] / (1.5 * keV_to_erg * (n_p + n_B))
    Te = Y[1] / (1.5 * keV_to_erg * n_e)
    sv = np.interp(np.clip(Ti, Ti_grid[0], Ti_grid[-1]), Ti_grid, sv_x)
    Pfus = n_p * n_B * sv * Q_erg
    Zeff = (n_p + n_B * Z_B**2 + n_a * 4) / n_e
    x = Te / E_rest_keV
    gg = (1 + 1.78 * x**1.34) + 2.12 * x * (1 + 1.1 * x + x**2
                                            - 1.25 * x**2.5) / Zeff
    Pbrems = 5.34e-31 * n_e**2 * Zeff * np.sqrt(Te) * gg * 1e7
    inst = Pfus / np.maximum(Pbrems, 1e-30)
    # transient window: time while Te < 0.9*Ti
    win_mask = Te < 0.9 * Ti
    win_dur = tt[win_mask][-1] if np.any(win_mask) else 0.0
    return dict(G=G, tau_ie=tau_ie, E_spark=E_spark, E_fus=E_fus,
                E_brems=E_brems, E_trans=E_trans, win_dur=win_dur,
                max_inst=float(np.max(inst)), tt=tt, Ti=Ti, Te=Te, n_a=n_a,
                Pfus=Pfus, Pbrems=Pbrems, t_end=sol.t[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ti-points', type=int, default=12)
    ap.add_argument('--taue-points', type=int, default=14)
    ap.add_argument('--te-ratio', type=float, default=0.1)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(_THIS, '..', 'figures')))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Ti_grid, SV, Te_grid, R_grid = build_interpolators()

    Ti_axis = np.linspace(200.0, 800.0, args.ti_points)
    tauE_axis = np.geomspace(0.01, 100.0, args.taue_points)

    # ---- representative time series ----
    reps = [('wang', 400.0, 1.0, 0.0), ('wang', 600.0, 10.0, 0.0)]
    fig, axes = plt.subplots(len(reps), 2, figsize=(13, 4.6 * len(reps)),
                             squeeze=False)
    for i, (xs, Ti0, tauE, ech) in enumerate(reps):
        r = run_pulse(SV[xs], Ti_grid, R_grid, Te_grid, Ti0, args.te_ratio,
                      tauE, eta_ch=ech)
        ax = axes[i][0]
        ax.plot(r['tt'], r['Ti'], 'r-', lw=2, label='$T_i$')
        ax.plot(r['tt'], r['Te'], 'b-', lw=2, label='$T_e$')
        ax.set_xlabel('t (s)'); ax.set_ylabel('T (keV)')
        ax.set_title(f'{xs}  $T_{{i,0}}$={Ti0:.0f}, $\\tau_E$={tauE:g}s, '
                     f'$\\tau_{{ie}}$={r["tau_ie"]:.2f}s  →  G={r["G"]:.3f}')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax = axes[i][1]
        ax.semilogy(r['tt'], r['Pfus'], 'r-', lw=2, label='$P_{fus}$')
        ax.semilogy(r['tt'], r['Pbrems'], 'b-', lw=2, label='$P_{brems}$')
        ax.set_xlabel('t (s)'); ax.set_ylabel('power (erg/s/cm³)')
        ax.set_title(f'transient window (Te<0.9Ti): {r["win_dur"]:.2f}s ; '
                     f'max inst $P_{{fus}}/P_{{brems}}$={r["max_inst"]:.2f}')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
    fig.suptitle('Pulsed 0-D: representative trajectories (spark = instantaneous '
                 f'hot-ion, $T_{{e,0}}$={args.te_ratio}$T_{{i,0}}$)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_ts = os.path.join(args.out, 'pulsed_timeseries.png')
    fig.savefig(out_ts, dpi=300, bbox_inches='tight')
    fig.savefig(out_ts.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig)

    # ---- G maps ----
    cases = [('wang', 0.0), ('TB', 0.0), ('wang', 0.3), ('TB', 0.3)]
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11))
    grid_rows = []
    summary = []
    for ax, (xs, ech) in zip(axes2.ravel(), cases):
        Gmap = np.zeros((len(tauE_axis), len(Ti_axis)))
        tauie_row = np.zeros(len(Ti_axis))
        for b, Ti0 in enumerate(Ti_axis):
            for a, tauE in enumerate(tauE_axis):
                r = run_pulse(SV[xs], Ti_grid, R_grid, Te_grid, Ti0,
                              args.te_ratio, tauE, eta_ch=ech)
                Gmap[a, b] = r['G']
                tauie_row[b] = r['tau_ie']
                grid_rows.append([xs, ech, f'{Ti0:.0f}', f'{tauE:.4f}',
                                  f'{r["tau_ie"]:.3f}', f'{r["G"]:.5f}',
                                  f'{r["max_inst"]:.3f}', f'{r["win_dur"]:.3f}'])
        lev = np.linspace(0, 2, 21)
        cf = ax.contourf(Ti_axis, tauE_axis, np.clip(Gmap, 0, 2), levels=lev,
                         cmap='RdYlGn', extend='max')
        ax.contour(Ti_axis, tauE_axis, Gmap, levels=[1.0], colors='k', linewidths=3)
        ax.plot(Ti_axis, tauie_row, 'c--', lw=1.6, label=r'$\tau_{ie}$')
        ax.axhspan(*TAU_E_BAND, color='blue', alpha=0.12)
        ax.axhline(TAU_E_ACHIEVABLE, color='blue', ls=':', lw=1.5,
                   label='achievable $\\tau_E$ ~1s')
        ax.set_yscale('log'); ax.set_xlabel('$T_{i,0}$ (keV)')
        ax.set_ylabel('$\\tau_E$ (s)')
        ax.set_title(f'{xs}, $\\eta_{{ch}}$={ech} — G (black=G=1)')
        ax.legend(fontsize=8, loc='upper left')
        plt.colorbar(cf, ax=ax, label='G = E_fus/(E_spark+E_brems+E_trans)')
        # island detection within achievable tau_E
        ach = tauE_axis <= TAU_E_ACHIEVABLE
        sub = Gmap[ach, :]
        isl = bool(np.any(sub > 1.0))
        gmax_ach = float(np.max(sub))
        gmax_all = float(np.max(Gmap))
        # smallest tau_E giving G>1 (any Ti)
        tauE_for_G1 = np.inf
        for a, tauE in enumerate(tauE_axis):
            if np.any(Gmap[a, :] > 1.0):
                tauE_for_G1 = tauE
                break
        summary.append([xs, ech, isl, f'{gmax_ach:.4f}', f'{gmax_all:.4f}',
                        ('%.3f' % tauE_for_G1) if np.isfinite(tauE_for_G1) else 'none'])
    fig2.suptitle('Pulsed 0-D net energy gain G over the cycle. Blue dotted = '
                  'achievable $\\tau_E$~1s; cyan = $\\tau_{ie}$. '
                  'G>1 above the achievable line = perfect-confinement artifact.',
                  fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    out_g = os.path.join(args.out, 'pulsed_Gmap.png')
    fig2.savefig(out_g, dpi=300, bbox_inches='tight')
    fig2.savefig(out_g.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig2)

    with open(os.path.join(args.out, 'pulsed_Gmap.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'eta_ch', 'Ti_init', 'tau_E_s', 'tau_ie_s',
                    'G', 'max_inst_Pfus_Pbrems', 'transient_window_s'])
        w.writerows(grid_rows)
    with open(os.path.join(args.out, 'pulsed_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'eta_ch', 'G>1_in_achievable_tauE(<=1s)',
                    'maxG_achievable', 'maxG_all_tauE', 'min_tauE_for_G1_s'])
        w.writerows(summary)

    print("=" * 80)
    print("PULSED 0-D SUMMARY  (achievable tau_E <= 1 s)")
    print("=" * 80)
    print(f"{'xsec':<6}{'eta_ch':<8}{'G>1 achievable?':<18}{'maxG(<=1s)':>12}"
          f"{'maxG(all)':>12}{'min_tauE(G=1)':>15}")
    for s in summary:
        print(f"{s[0]:<6}{s[1]:<8}{('YES' if s[2] else 'no'):<18}{s[3]:>12}"
              f"{s[4]:>12}{s[5]:>15}")
    print("-" * 80)
    print(f"[OK] {out_ts}\n[OK] {out_g}")
    print("Note: G>1 only at tau_E >> ~1 s is a perfect-confinement artifact "
          "(Lawson p-11B tau_E ~10-100 s is NOT achievable at 1e14).")


if __name__ == '__main__':
    main()
