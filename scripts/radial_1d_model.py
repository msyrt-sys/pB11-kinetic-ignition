#!/usr/bin/env python3
"""
radial_1d_model.py — 1-D radial (cylindrical) steady-state p-11B power balance.

Tests the one thing 0-D cannot see: SPATIAL decoupling. Can pushing the
bremsstrahlung/ash to a separate (cold/edge) region from the hot fusion core
open a positive net-power island? (Ochs-Fisch-style ash/heat separation.)

Prescribed-profile budget-level model (cylinder, per unit length, r in [0,a]):
  Ti(r), Te(r)=tau(r)*Ti(r), n_alpha(r) prescribed (hot core / cold-ashy edge);
  n_p, n_B flat. Local P_fus(r), P_brems(r), P_ie(r) from the existing 0-D
  formulas evaluated pointwise.

  P_net = INT P_fus dV - INT P_brems dV - P_cond_edge

*** SPATIAL-RIDER HONESTY HOOK ***
  Holding a hot core against a cold edge is NOT free: the temperature gradient
  drives a conductive heat flux that escapes at the boundary.
    P_cond_edge = -2*pi*a*chi*(n_i dTi/dr + n_e dTe/dr)|_{r=a}   (keV->erg)
  chi = heat diffusivity. REALISTIC scaling: Bohm chi_B[cm^2/s]=T[keV]*6.25e5/B
  and gyro-Bohm chi_gB = chi_B*(rho_p/a), B in [5,20] T. chi_i=chi_e (one chi at
  edge). chi->0 (perfect insulation) gives FAKE gain (flagged), like tau_E->inf
  in the pulsed model.

  *** Reported prominently: P_transport / INT P_fus at realistic chi. ***

CAVEATS (optimistic-biased -> real result WORSE):
  ** Ash-particle transport NOT charged. Moving ash to the edge is NOT free:
     it costs particle transport AND the channeling wave gives alpha energy to
     electrons en route. This model charges only HEAT conduction, so it is
     biased IN FAVOR of ash separation. Any positive island => real one worse. **
  * Local Te<Ti decoupling (low core tau) has its own 0-D e-i recirculation cost
    (Rider, shown net-negative earlier); only the SPATIAL transport is charged here.
  * Flat density profile; prescribed (not transport-self-consistent) profiles.

SELF-CHECK (mandatory): (a) <sigma v>(Ti)/R(Te) interpolation <0.1%; (b) a
UNIFORM profile (no gradient -> P_cond=0) must reproduce the 0-D ignition_check
(P_F - P_Brem) to <1%.

Outputs: radial_profiles.png, radial_Pnet_maps.png, radial_grid.csv, radial_summary.csv
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
from collision_operators import m_p_g, m_B_g, m_e_g, e_esu, c_cm, Z_B  # noqa: E402
from power_balance import E_rest_keV  # noqa: E402
sys.stdout = _real

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402

keV_to_erg = cs.keV_to_erg
Q_erg = cs.Q_pB11_keV * keV_to_erg
A_MINOR = 100.0          # cm, minor radius
N_R = 300
LNL = 17.0


def build_interp():
    Ti_grid = np.linspace(50.0, 650.0, 1201)
    Te_grid = np.linspace(5.0, 650.0, 1291)
    SV = {}
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        SV[xs] = cs.sigma_v_TB_numerical(Ti_grid)
    print("=" * 70)
    print("INTERPOLATION SELF-CHECK (< 0.1%)")
    print("=" * 70)
    ok = True
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        for T in (177.7, 288.3, 401.1, 522.2):
            full = cs.sigma_v_TB_numerical(T)[0]
            ip = float(np.interp(T, Ti_grid, SV[xs]))
            e = abs(ip - full) / full * 100
            ok &= e < 0.1
            print(f"  <sv> {xs:<4} Ti={T:6.1f}: err={e:.4f}%  {'OK' if e<0.1 else 'FAIL'}")
    if not ok:
        print("FAIL — abort"); sys.exit(2)
    print("SELF-CHECK (interp) PASSED.\n")
    return Ti_grid, SV


def brems_ergs(n_e, Te, Zeff):
    Pcl = 5.34e-31 * n_e**2 * Zeff * np.sqrt(Te)
    x = Te / E_rest_keV
    g = (1 + 1.78 * x**1.34) + 2.12 * x * (1 + 1.1 * x + x**2 - 1.25 * x**2.5) / Zeff
    return Pcl * g * 1e7                      # erg/s/cm3


def chi_bohm(T_keV, B_T):
    return T_keV * 6.25e5 / B_T               # cm^2/s


def rho_p(T_keV, B_T):
    v = np.sqrt(2 * T_keV * keV_to_erg / m_p_g)
    omega_c = e_esu * (B_T * 1e4) / (m_p_g * c_cm)
    return v / omega_c                         # cm


def profiles(Ti_core, kappa_T, nu_T, tau_core, tau_edge, n_alpha_avg,
             ash_edge, f_B=0.15, n_i=1e14, N=N_R):
    r = np.linspace(0, A_MINOR, N)
    x = r / A_MINOR
    Ti_edge = kappa_T * Ti_core
    Ti = Ti_edge + (Ti_core - Ti_edge) * (1 - x**2)**nu_T
    # Te as a directly MONOTONIC profile (core low tau, edge high tau): core
    # Te = tau_core*Ti_core, edge Te = tau_edge*Ti_edge. Guarantees dTe/dr <= 0.
    Te_core = tau_core * Ti_core
    Te_edge = tau_edge * Ti_edge
    Te = Te_edge + (Te_core - Te_edge) * (1 - x**2)**nu_T
    n_p = (1 - f_B) * n_i * np.ones_like(r)
    n_B = f_B * n_i * np.ones_like(r)
    # ash: edge-weighted shape normalised to volume-average n_alpha_avg
    w = 1.0 + ash_edge * x**4
    vol_w = np.trapezoid(w * r, r) / np.trapezoid(r, r)
    n_alpha = n_alpha_avg * w / vol_w
    return r, x, Ti, Te, n_p, n_B, n_alpha


def evaluate(xs, Ti_grid, SV, Ti_core, kappa_T, nu_T, tau_core, tau_edge,
             n_alpha_avg, ash_edge, B_T, chi_model):
    cs.CROSS_SECTION = xs
    r, x, Ti, Te, n_p, n_B, n_alpha = profiles(Ti_core, kappa_T, nu_T,
                                               tau_core, tau_edge, n_alpha_avg,
                                               ash_edge)
    n_e = n_p + Z_B * n_B + 2 * n_alpha
    n_i = n_p + n_B + n_alpha
    Zeff = (n_p + n_B * Z_B**2 + n_alpha * 4) / n_e
    sv = np.interp(np.clip(Ti, Ti_grid[0], Ti_grid[-1]), Ti_grid, SV[xs])
    Pfus = n_p * n_B * sv * Q_erg                          # erg/s/cm3
    Pbrems = brems_ergs(n_e, Te, Zeff)
    # volume integrals (cylindrical, per unit length): INT f 2 pi r dr
    twopir = 2 * np.pi * r
    Ifus = np.trapezoid(Pfus * twopir, r)
    Ibrems = np.trapezoid(Pbrems * twopir, r)
    # conduction loss at the edge (gradient-driven)
    T_loc = 0.5 * (Ti + Te)
    chi_r = chi_bohm(T_loc, B_T)              # local Bohm diffusivity (cm^2/s)
    if chi_model == 'gyro':
        chi_r = chi_r * rho_p(T_loc, B_T) / A_MINOR
    dTi_dr = np.gradient(Ti, r)
    dTe_dr = np.gradient(Te, r)
    # outward conductive power crossing radius r (core->edge heat loss)
    Pcross = -2 * np.pi * r * chi_r * (n_i * dTi_dr + n_e * dTe_dr) * keV_to_erg
    # transport loss = PEAK outward conductive power (the bottleneck the core
    # heat must cross to be removed). >=0 for monotonic T; 0 for uniform.
    P_cond = float(np.max(Pcross))
    P_net = Ifus - Ibrems - P_cond
    T_avg = np.trapezoid(T_loc * twopir, r) / np.trapezoid(twopir, r)
    chi = float(np.mean(chi_r))
    return dict(r=r, Ti=Ti, Te=Te, n_alpha=n_alpha, Pfus=Pfus, Pbrems=Pbrems,
                Pcross=Pcross, Ifus=Ifus, Ibrems=Ibrems, P_cond=P_cond,
                P_net=P_net, chi=chi, T_avg=T_avg,
                ratio=P_net / Ibrems if Ibrems > 0 else -np.inf,
                transp_over_fus=P_cond / Ifus if Ifus > 0 else np.inf)


def self_check_0d(Ti_grid, SV):
    print("=" * 70)
    print("SELF-CHECK: uniform 1-D profile  ==  0-D ignition_check  (< 1%)")
    print("=" * 70)
    ok = True
    n_i = 1e14; f_B = 0.15
    n_p = (1 - f_B) * n_i; n_B = f_B * n_i
    for xs in ('wang', 'TB'):
        for Ti0, Te0 in ((300., 150.), (450., 200.)):
            # uniform: kappa_T=1 (no Ti gradient), tau uniform -> Te uniform,
            # n_alpha=0 ; gradient 0 -> P_cond=0
            res = evaluate(xs, Ti_grid, SV, Ti0, 1.0, 1.0, Te0 / Ti0, Te0 / Ti0,
                           0.0, 0.0, 10.0, 'bohm')
            V = np.pi * A_MINOR**2
            pf_1d = res['Ifus'] / V / 1e7                  # W/cm3
            pb_1d = res['Ibrems'] / V / 1e7
            cs.CROSS_SECTION = xs
            info = pb.ignition_check(n_p, n_B, Ti0, T_e_keV=Te0)
            ef = abs(pf_1d - info['P_F']) / info['P_F'] * 100
            eb = abs(pb_1d - info['P_Brem']) / info['P_Brem'] * 100
            pc = abs(res['P_cond'])
            ok &= ef < 1 and eb < 1 and pc < 1e-3 * abs(res['Ifus'])
            print(f"  {xs:<4} Ti={Ti0:.0f},Te={Te0:.0f}: "
                  f"P_F 1d={pf_1d:.4e} 0d={info['P_F']:.4e} ({ef:.3f}%); "
                  f"P_B 1d={pb_1d:.4e} 0d={info['P_Brem']:.4e} ({eb:.3f}%); "
                  f"P_cond/Ifus={pc/abs(res['Ifus']):.1e}")
    print("SELF-CHECK (uniform->0D):", "PASSED" if ok else "FAILED")
    if not ok:
        sys.exit(2)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ti-points', type=int, default=13)
    ap.add_argument('--b-points', type=int, default=12)
    ap.add_argument('--kappa-T', type=float, default=0.3)
    ap.add_argument('--nu-T', type=float, default=1.5)
    ap.add_argument('--tau-core', type=float, default=0.4)
    ap.add_argument('--tau-edge', type=float, default=0.9)
    ap.add_argument('--n-alpha', type=float, default=0.02 * 1.6e14)
    ap.add_argument('--ash-edge', type=float, default=3.0)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(_THIS, '..', 'figures')))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Ti_grid, SV = build_interp()
    self_check_0d(Ti_grid, SV)

    # ---- representative radial profiles ----
    rep = evaluate('wang', Ti_grid, SV, 400.0, args.kappa_T, args.nu_T,
                   args.tau_core, args.tau_edge, args.n_alpha, args.ash_edge,
                   10.0, 'bohm')
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    xr = rep['r'] / A_MINOR
    ax[0].plot(xr, rep['Ti'], 'r-', lw=2, label='$T_i$')
    ax[0].plot(xr, rep['Te'], 'b-', lw=2, label='$T_e$')
    ax[0].set_xlabel('r/a'); ax[0].set_ylabel('T (keV)'); ax[0].legend()
    ax[0].set_title('hot core / cold edge profile'); ax[0].grid(alpha=0.3)
    ax2 = ax[0].twinx()
    ax2.plot(xr, rep['n_alpha'] / 1e12, 'g--', lw=1.5)
    ax2.set_ylabel('$n_\\alpha$ (10¹² cm⁻³)', color='g')
    ax[1].semilogy(xr, np.asarray(rep['Pfus']) / 1e7, 'r-', lw=2, label='$P_\\mathrm{fus}(r)$')
    ax[1].semilogy(xr, np.asarray(rep['Pbrems']) / 1e7, 'b-', lw=2, label='$P_\\mathrm{brems}(r)$')
    ax[1].set_xlabel('r/a'); ax[1].set_ylabel('W/cm$^3$'); ax[1].legend()
    ax[1].set_title('local power densities'); ax[1].grid(alpha=0.3, which='both')
    cumfus = np.array([np.trapezoid(rep['Pfus'][:i + 1] * 2 * np.pi * rep['r'][:i + 1],
                                    rep['r'][:i + 1]) for i in range(len(rep['r']))])
    ax[2].plot(xr, np.abs(rep['Pcross']) / 1e7, 'm-', lw=2,
               label='conductive power crossing r')
    ax[2].plot(xr, cumfus / 1e7, 'r--', lw=2, label='cumulative $P_\\mathrm{fus}$ inside r')
    ax[2].set_yscale('log'); ax[2].set_xlabel('r/a'); ax[2].set_ylabel('W/cm')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3, which='both')
    ax[2].set_title(f'transport vs fusion  '
                    f'($P_\\mathrm{{cond}}/\\!\\int\\! P_\\mathrm{{fus}}$ = '
                    f'{rep["transp_over_fus"]:.0f}$\\times$ @ Bohm, B=10 T)')
    fig.suptitle('Representative decoupled profile (Wang, $T_{i,core}$=400 keV, '
                 f'$\\kappa_T$={args.kappa_T}, Bohm $\\chi$, B=10T)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_p = os.path.join(args.out, 'radial_profiles.png')
    fig.savefig(out_p, dpi=300, bbox_inches='tight')
    fig.savefig(out_p.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig)

    # ---- P_net and P_transport/P_fus maps ----
    Ti_axis = np.linspace(200.0, 600.0, args.ti_points)
    B_axis = np.linspace(5.0, 20.0, args.b_points)
    cases = [('wang', 'bohm'), ('wang', 'gyro'), ('TB', 'bohm'), ('TB', 'gyro')]
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 9))
    grid_rows = []
    summary = []
    for j, (xs, chim) in enumerate(cases):
        Rnet = np.zeros((len(B_axis), len(Ti_axis)))
        Rtr = np.zeros((len(B_axis), len(Ti_axis)))
        Pnet0 = np.zeros(len(Ti_axis))   # chi->0 reference (fake gain)
        for b, Bv in enumerate(B_axis):
            for a, Ti0 in enumerate(Ti_axis):
                res = evaluate(xs, Ti_grid, SV, Ti0, args.kappa_T, args.nu_T,
                               args.tau_core, args.tau_edge, args.n_alpha,
                               args.ash_edge, Bv, chim)
                Rnet[b, a] = res['ratio']
                Rtr[b, a] = res['transp_over_fus']
                grid_rows.append([xs, chim, f'{Ti0:.0f}', f'{Bv:.1f}',
                                  f'{res["P_net"]:.4e}', f'{res["ratio"]:.4f}',
                                  f'{res["transp_over_fus"]:.2f}'])
        # chi->0 reference: P_net = Ifus - Ibrems (no transport)
        for a, Ti0 in enumerate(Ti_axis):
            res0 = evaluate(xs, Ti_grid, SV, Ti0, args.kappa_T, args.nu_T,
                            args.tau_core, args.tau_edge, args.n_alpha,
                            args.ash_edge, 1e9, chim)  # B huge -> chi ~ 0
            Pnet0[a] = (res0['Ifus'] - res0['Ibrems']) / res0['Ibrems']

        axA = axes2[0][j]
        lev = np.linspace(-1, 1, 21)
        cf = axA.contourf(Ti_axis, B_axis, np.clip(Rnet, -1, 1), levels=lev,
                          cmap='RdYlGn', extend='both')
        if np.any(Rnet > 0):
            axA.contour(Ti_axis, B_axis, Rnet, levels=[0], colors='k', linewidths=3)
        axA.set_xlabel('$T_{i,core}$ (keV)'); axA.set_ylabel('B (T)')
        axA.set_title(f'{xs}, {chim}-Bohm: $P_\\mathrm{{net}}/\\!\\int\\! P_\\mathrm{{brems}}$')
        plt.colorbar(cf, ax=axA)
        axB = axes2[1][j]
        cfb = axB.contourf(Ti_axis, B_axis, np.log10(np.clip(Rtr, 1, 1e7)),
                           levels=np.linspace(0, 6, 25), cmap='inferno')
        axB.set_xlabel('$T_{i,core}$ (keV)'); axB.set_ylabel('B (T)')
        axB.set_title(f'{xs}, {chim}: $\\log_{{10}}(P_\\mathrm{{transport}}/\\!\\int\\! P_\\mathrm{{fus}})$')
        plt.colorbar(cfb, ax=axB, label='$\\log_{10}$ ratio')

        isl = bool(np.any(Rnet > 0))
        minRtr = float(np.min(Rtr))
        summary.append([xs, chim, isl, f'{np.max(Rnet):.4f}', f'{minRtr:.1f}',
                        f'{np.max(Pnet0):.4f}'])
    fig2.suptitle('1-D radial: net power (top) and transport/fusion ratio (bottom). '
                  'Realistic Bohm/gyro-Bohm $B\\in[5,20]$ T. '
                  '$P_\\mathrm{transport} \\gg P_\\mathrm{fus}$ (bottom, log scale): '
                  'spatial decoupling fails.',
                  fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    out_m = os.path.join(args.out, 'radial_Pnet_maps.png')
    fig2.savefig(out_m, dpi=300, bbox_inches='tight')
    fig2.savefig(out_m.replace('.png', '.pdf'), bbox_inches='tight'); plt.close(fig2)

    with open(os.path.join(args.out, 'radial_grid.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'chi_model', 'Ti_core', 'B_T', 'P_net_erg_s',
                    'Pnet_over_Pbrems', 'Ptransport_over_Pfus'])
        w.writerows(grid_rows)
    with open(os.path.join(args.out, 'radial_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'chi_model', 'Pnet>0_realistic_B',
                    'max_Pnet_over_Pbrems', 'min_Ptransport_over_Pfus',
                    'max_Pnet_chi->0_fake'])
        w.writerows(summary)

    print("=" * 80)
    print("1-D RADIAL SUMMARY (decoupled profile, realistic B in [5,20] T)")
    print("=" * 80)
    print(f"{'xsec':<6}{'chi':<7}{'Pnet>0?':<9}{'maxPnet/Pb':>12}"
          f"{'min Ptr/Pfus':>14}{'chi->0 fake':>13}")
    for s in summary:
        print(f"{s[0]:<6}{s[1]:<7}{('YES' if s[2] else 'no'):<9}{s[3]:>12}"
              f"{s[4]:>14}{s[5]:>13}")
    print("-" * 80)
    print(f"[OK] {out_p}\n[OK] {out_m}")
    print("min Ptr/Pfus = smallest transport/fusion ratio over B∈[5,20] "
          "(>>1 means transport buries fusion). 'chi->0 fake' = decoupled "
          "Ifus-Ibrems with NO transport (perfect-insulation artifact).")
    print("CAVEAT: ash-particle transport NOT charged -> biased FAVOURING "
          "decoupling; real result is worse.")


if __name__ == '__main__':
    main()
