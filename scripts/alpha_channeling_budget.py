#!/usr/bin/env python3
"""
alpha_channeling_budget.py — budget-level net-power map for p-11B alpha channeling.

Boundary study (NOT a feasibility proof): does a positive net-power island exist
for p-11B thermonuclear ignition under alpha channeling, and in which parameter
window? Four-term budget, NOTHING dropped:

    P_net = P_fus(tail-enhanced) - P_brems(ash-cleaned) - P_drive - P_relax

with the kinetic enhancement DERIVED from channeling (not a free n_alpha x F).

Axes:
  eta_ch  in [0, 1]        : alpha-channeling efficiency (fraction of alpha
                             power redirected by the wave)
  r = n_alpha/n_e0 in [0.005, 0.10] : steady-state ash density set by the
                             extraction rate f_extract (n_alpha = S_alpha*tau_s
                             / f_extract; r~0.05 = no active removal, r~0.01 =
                             Ochs active removal). Reported as the physical axis.
  tau = Te/Ti in [0.3, 1.0]: electron-ion decoupling.

Three (f_tail, eta_drive) scenarios: optimistic (0.6, 0.8), nominal (0.3, 0.5),
pessimistic (0.15, 0.3). Both cross sections (Wang, TB).

TWO P_relax variants (run side by side):
  (i)  no-double-count : decoupling cost = max(0, P_alpha_e + P_ie - P_brems),
       the electron-power EXCESS over bremsstrahlung (zero at the self-consistent
       Te; positive only if Te is forced below what the electron balance supports).
  (ii) pure-Rider      : decoupling cost = full P_ie (the entire ion->electron
       collisional transfer treated as recirculating load).
Both variants ALSO charge the tail maintenance P_relax_tail = f_tail*P_ch (the
wave-driven tail thermalizes on the slowing-down time and must be continuously
re-driven) and the wave-drive inefficiency P_drive = (1/eta_drive - 1)*P_ch.

CAVEATS (optimistic-biased assumptions, must accompany any positive island):
  * <E_tail> fixed at the cross-section peak (600 keV CM). A real wave-driven
    tail spreads and loses energy collisionally, sampling lower-sigma energies,
    so this UPPER-BOUNDS the tail reactivity gain.
  * P_alpha taken as the THERMAL fusion power (no tail->more-alpha feedback).
  * Te treated as a free axis; the tail's own electron heating is not fed back
    into Te/brems.
Literature: alpha-channeling efficiency in idealized theory (Fisch-Rax PRL 69,
612 (1992); Ochs-Fisch, PRE 106, 055215 (2022), arXiv:2210.08076) ~0.3-0.6;
practical/demonstrated values are much lower, and Rider (PoP 1995, 1997) argues
the recirculating power makes non-equilibrium p-11B net-negative. The defensible
sub-region eta_ch <= 0.5 is marked on every panel.

Outputs (to --out, default ../figures):
  - alpha_channeling_island.png    : (eta_ch x n_alpha) at self-consistent Te,
                                     2 (xsec) x 3 (scenario), P_net=0 for i & ii
  - alpha_channeling_tau.png       : (eta_ch x tau) decoupling slice (Wang nominal)
  - alpha_channeling_grid.csv      : full 3D budget grid
  - alpha_channeling_summary.csv   : positive-island summary per (xsec,scenario,variant)
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
from collision_operators import m_p_g, m_B_g, Z_B   # noqa: E402
sys.stdout = _real

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402

keV_to_erg = cs.keV_to_erg
barn_cm2 = cs.barn_cm2
Q_erg = cs.Q_pB11_keV * keV_to_erg
E_CM_PEAK = 600.0                          # keV, reactivity peak (CM)
E_TAIL_LAB = E_CM_PEAK * (m_p_g + m_B_g) / m_B_g   # ~654 keV lab kinetic energy

SCENARIOS = {
    'optimistic':  dict(f_tail=0.60, eta_drive=0.80),
    'nominal':     dict(f_tail=0.30, eta_drive=0.50),
    'pessimistic': dict(f_tail=0.15, eta_drive=0.30),
}
TAU_SC = 0.40           # representative self-consistent Te/Ti (band 0.34-0.52)
TAU_BAND = (0.34, 0.52)
ETA_DEFENSIBLE = 0.5    # idealized-theory upper bound for eta_ch


def budget(sigma_name, Ti, tau, eta_ch, r_ash, f_tail, eta_drive):
    """Return dict with all four budget terms (W/cm^3) for both relax variants."""
    cs.CROSS_SECTION = sigma_name
    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i
    n_e0 = n_p + Z_B * n_B
    n_alpha = r_ash * n_e0
    n_e = n_e0 + 2.0 * n_alpha
    Te = tau * Ti

    # base fusion / alpha power
    P_fus_th = float(pb.P_fusion_thermal(n_p, n_B, Ti))
    P_alpha = P_fus_th
    P_ch = eta_ch * P_alpha

    # tail enhancement (derived from channeled power)
    tau_s = als.slowing_down_time_alpha(n_e, Te)      # A/Z^2 same for p and alpha
    P_ch_tail_erg = f_tail * P_ch * 1e7               # W/cm3 -> erg/s/cm3
    n_tail = P_ch_tail_erg * tau_s / (E_TAIL_LAB * keV_to_erg)
    v_tail = np.sqrt(2 * E_TAIL_LAB * keV_to_erg / m_p_g)
    sigma_cm2 = cs.sigma_fusion(E_CM_PEAK)[0] * barn_cm2
    R_tail = n_tail * n_B * sigma_cm2 * v_tail
    dP_fus = R_tail * Q_erg / 1e7                     # W/cm3
    P_fus = P_fus_th + dP_fus

    # bremsstrahlung (ash-cleaned, operating Te)
    P_brems = float(pb.P_brem_with_ash(n_p, n_B, n_alpha, Te))

    # wave-drive inefficiency
    P_drive = (1.0 / eta_drive - 1.0) * P_ch

    # relaxation
    P_relax_tail = f_tail * P_ch
    P_ie = float(pb.P_ion_electron_transfer(n_p, n_B, Ti, Ti, Te))
    frac_e = pb.alpha_power_to_electrons_fraction(Te, n_e, n_p, n_B)
    P_alpha_e = (1 - eta_ch) * P_alpha * frac_e
    relax_dec_i = max(0.0, P_alpha_e + P_ie - P_brems)
    relax_dec_ii = P_ie

    net_i = P_fus - P_brems - P_drive - P_relax_tail - relax_dec_i
    net_ii = P_fus - P_brems - P_drive - P_relax_tail - relax_dec_ii
    return dict(P_fus=P_fus, P_fus_th=P_fus_th, dP_fus=dP_fus, P_brems=P_brems,
                P_drive=P_drive, P_relax_tail=P_relax_tail, P_ie=P_ie,
                relax_dec_i=relax_dec_i, relax_dec_ii=relax_dec_ii,
                net_i=net_i, net_ii=net_ii, n_alpha=n_alpha, n_e=n_e)


def best_over_Ti(sigma_name, tau, eta_ch, r_ash, f_tail, eta_drive, variant,
                 T_grid):
    """Max P_net/P_brems over Ti; returns (ratio, Ti_opt, budget_dict)."""
    key = 'net_i' if variant == 'i' else 'net_ii'
    best = (-np.inf, None, None)
    for Ti in T_grid:
        b = budget(sigma_name, Ti, tau, eta_ch, r_ash, f_tail, eta_drive)
        ratio = b[key] / b['P_brems'] if b['P_brems'] > 0 else -np.inf
        if ratio > best[0]:
            best = (ratio, Ti, b)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cross-sections', nargs='+', default=['wang', 'TB'],
                    choices=['TB', 'wang'])
    ap.add_argument('--eta-points', type=int, default=21)
    ap.add_argument('--r-points', type=int, default=18)
    ap.add_argument('--tau-points', type=int, default=15)
    ap.add_argument('--t-min', type=float, default=180.0)
    ap.add_argument('--t-max', type=float, default=500.0)
    ap.add_argument('--t-points', type=int, default=17)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(_THIS, '..', 'figures')))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eta_axis = np.linspace(0.0, 1.0, args.eta_points)
    r_axis = np.geomspace(0.005, 0.10, args.r_points)
    tau_axis = np.linspace(0.30, 1.0, args.tau_points)
    T_grid = np.linspace(args.t_min, args.t_max, args.t_points)

    # ---------- FIGURE 1: (eta_ch x r) at self-consistent Te ----------
    fig1, axes1 = plt.subplots(len(args.cross_sections), 3,
                               figsize=(16, 4.6 * len(args.cross_sections)),
                               squeeze=False)
    grid_rows = []
    summary = []

    for ic, xs in enumerate(args.cross_sections):
        for js, sc in enumerate(['optimistic', 'nominal', 'pessimistic']):
            f_tail = SCENARIOS[sc]['f_tail']
            eta_drive = SCENARIOS[sc]['eta_drive']
            ratio_i = np.zeros((len(r_axis), len(eta_axis)))
            ratio_ii = np.zeros((len(r_axis), len(eta_axis)))
            for a, r in enumerate(r_axis):
                for b, eta in enumerate(eta_axis):
                    ri, Ti_i, _ = best_over_Ti(xs, TAU_SC, eta, r, f_tail,
                                               eta_drive, 'i', T_grid)
                    rii, Ti_ii, _ = best_over_Ti(xs, TAU_SC, eta, r, f_tail,
                                                 eta_drive, 'ii', T_grid)
                    ratio_i[a, b] = ri
                    ratio_ii[a, b] = rii
                    grid_rows.append([xs, sc, f'{eta:.3f}', f'{r:.4f}',
                                      f'{TAU_SC:.2f}', f'{Ti_i:.0f}',
                                      f'{ri:.4f}', f'{rii:.4f}'])

            ax = axes1[ic][js]
            lev = np.linspace(-0.6, 0.6, 25)
            cf = ax.contourf(eta_axis, r_axis, np.clip(ratio_i, -0.6, 0.6),
                             levels=lev, cmap='RdYlGn', extend='both')
            ax.contour(eta_axis, r_axis, ratio_i, levels=[0.0], colors='k',
                       linewidths=2.5)
            ax.contour(eta_axis, r_axis, ratio_ii, levels=[0.0], colors='k',
                       linewidths=2.0, linestyles='--')
            ax.axvline(ETA_DEFENSIBLE, color='blue', ls=':', lw=1.5)
            ax.set_yscale('log')
            ax.set_xlabel(r'$\eta_{\rm ch}$ (channeling efficiency)', fontsize=10)
            ax.set_ylabel(r'$n_\alpha/n_{e0}$ (ash)', fontsize=10)
            ax.set_title(f'{xs} — {sc}\n(f_tail={f_tail}, '
                         f'eta_drive={eta_drive})', fontsize=10)
            plt.colorbar(cf, ax=ax, label='$P_{net}/P_{brems}$')

            # island detection in defensible region (eta<=0.5)
            for vi, rg in [('i', ratio_i), ('ii', ratio_ii)]:
                mask = eta_axis <= ETA_DEFENSIBLE
                sub = rg[:, mask]
                island = bool(np.any(sub > 0))
                if island:
                    ia, ib = np.unravel_index(np.argmax(sub), sub.shape)
                    corner = (eta_axis[mask][ib], r_axis[ia], sub[ia, ib])
                else:
                    corner = (None, None, float(np.max(sub)))
                summary.append([xs, sc, vi, 'Te=self-consistent', island,
                                corner[0], corner[1], f'{corner[2]:.4f}'])

    fig1.suptitle('p-$^{11}$B alpha-channeling net-power island '
                  f'($T_e/T_i$={TAU_SC}; solid = $P_{{net}}$=0 variant i, '
                  'dashed = variant ii; dotted blue = defensible $\\eta_{ch}$=0.5)',
                  fontsize=12)
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = os.path.join(args.out, 'alpha_channeling_island.png')
    fig1.savefig(out1, dpi=135, bbox_inches='tight')
    plt.close(fig1)

    # ---------- FIGURE 2: (eta_ch x tau) decoupling slice ----------
    r_fix = 0.02
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2))
    for k, vi in enumerate(['i', 'ii']):
        f_tail = SCENARIOS['nominal']['f_tail']
        eta_drive = SCENARIOS['nominal']['eta_drive']
        Z = np.zeros((len(tau_axis), len(eta_axis)))
        Zother = np.zeros((len(tau_axis), len(eta_axis)))
        for a, tau in enumerate(tau_axis):
            for b, eta in enumerate(eta_axis):
                ri, _, _ = best_over_Ti('wang', tau, eta, r_fix, f_tail,
                                        eta_drive, vi, T_grid)
                ro, _, _ = best_over_Ti('wang', tau, eta, r_fix, f_tail,
                                        eta_drive, 'ii' if vi == 'i' else 'i',
                                        T_grid)
                Z[a, b] = ri
                Zother[a, b] = ro
        ax = axes2[k]
        lev = np.linspace(-0.6, 0.6, 25)
        cf = ax.contourf(eta_axis, tau_axis, np.clip(Z, -0.6, 0.6), levels=lev,
                         cmap='RdYlGn', extend='both')
        ax.contour(eta_axis, tau_axis, Z, levels=[0.0], colors='k', linewidths=2.5)
        ax.axhspan(*TAU_BAND, color='royalblue', alpha=0.18)
        ax.axvline(ETA_DEFENSIBLE, color='blue', ls=':', lw=1.5)
        ax.set_xlabel(r'$\eta_{\rm ch}$', fontsize=11)
        ax.set_ylabel(r'$\tau = T_e/T_i$', fontsize=11)
        ax.set_title(f'Wang, nominal scenario, $n_\\alpha/n_{{e0}}$={r_fix}\n'
                     f'P_relax variant ({vi})', fontsize=10)
        plt.colorbar(cf, ax=ax, label='$P_{net}/P_{brems}$')
    fig2.suptitle('Decoupling axis: shaded = self-consistent $\\tau$ band '
                  '(below it = forced cooling, penalized by P_relax)', fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = os.path.join(args.out, 'alpha_channeling_tau.png')
    fig2.savefig(out2, dpi=135, bbox_inches='tight')
    plt.close(fig2)

    # ---------- CSV ----------
    grid_csv = os.path.join(args.out, 'alpha_channeling_grid.csv')
    with open(grid_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'scenario', 'eta_ch', 'n_alpha_over_ne0',
                    'tau', 'Ti_opt', 'Pnet_Pbrems_variant_i',
                    'Pnet_Pbrems_variant_ii'])
        w.writerows(grid_rows)
    sum_csv = os.path.join(args.out, 'alpha_channeling_summary.csv')
    with open(sum_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'scenario', 'relax_variant', 'Te_mode',
                    'positive_island_in_defensible(eta<=0.5)', 'eta_ch_at_max',
                    'n_alpha_at_max', 'max_Pnet_Pbrems'])
        w.writerows(summary)

    # ---------- console summary ----------
    print("=" * 78)
    print("POSITIVE-ISLAND SUMMARY (defensible region eta_ch <= 0.5, Te self-consistent)")
    print("=" * 78)
    print(f"{'xsec':<6}{'scenario':<13}{'variant':<8}{'island?':<9}"
          f"{'eta*':>7}{'n_a*':>9}{'maxPnet/Pb':>12}")
    for row in summary:
        xs, sc, vi, _, island, eta_s, na_s, mx = row
        es = f'{eta_s:.2f}' if eta_s is not None else '  -'
        ns = f'{na_s:.3f}' if na_s is not None else '   -'
        print(f"{xs:<6}{sc:<13}{vi:<8}{('YES' if island else 'no'):<9}"
              f"{es:>7}{ns:>9}{mx:>12}")
    print("-" * 78)
    print(f"[OK] {out1}\n[OK] {out2}\n[OK] {grid_csv}\n[OK] {sum_csv}")


if __name__ == '__main__':
    main()
