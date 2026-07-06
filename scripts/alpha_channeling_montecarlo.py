#!/usr/bin/env python3
"""
alpha_channeling_montecarlo.py — full combinatorial (Sobol) net-power scan.

Frees SEVEN+ levers simultaneously and asks: does a positive net-power island
exist ANYWHERE in the defensible parameter space for p-11B alpha channeling?

Same four-term budget as alpha_channeling_budget.py (NOTHING dropped):
    P_net = P_fus(tail-enhanced) - P_brems(ash+impurity) - P_sync - P_drive - P_relax
with the kinetic enhancement DERIVED from channeling (eta_ch, f_tail), both
P_relax variants, and the honesty hooks preserved.

Sampled levers (Sobol, defensible ranges):
  Ti      in [150, 600] keV
  tau     in [0.30, 1.0]   (Te/Ti; P_relax penalises tau below self-consistent)
  f_B     in [0.05, 0.30]
  r_ash   in [0.001, 0.05] (n_alpha/n_e0)
  eta_ch  in [0, 0.5]      (idealized-theory ceiling; >0.5 not defensible)
  f_tail  in [0.10, 0.60]
  eta_drive in [0.30, 0.80]
  f_imp   in [0, 0.02]     (impurity electron fraction)
  xi_sync in [0, 0.30]     (synchrotron loss as fraction of P_brems)

Scenarios: Z_imp in {6 (carbon), 26 (iron, realistic wall)} (brems ~ Z^2, so
fixing a single low-Z would understate the impurity penalty). Cross sections:
Wang and TB, separately. P_relax variants: (i) no-double-count, (ii) pure-Rider.

n_i is NOT a lever (P_net/P_brems is density-independent: both ~ n^2).

MANDATORY SELF-CHECK (runs first, aborts on failure): the precomputed
<sigma v>(Ti) and R(Te) interpolators must match the full integrator to < 0.1%
at several off-grid points. If interpolation breaks the physics the whole scan
is void.

CAVEAT (optimistic-biased): <E_tail> fixed at the cross-section peak (600 keV
CM) upper-bounds the tail reactivity; P_alpha is the thermal fusion power
(no tail->alpha feedback); Te is a free axis.

Lit: Fisch-Rax PRL 69, 612 (1992); Ochs-Fisch PRE 106, 055215 (2022),
arXiv:2210.08076; Rider, Phys. Plasmas 2, 1853 (1995) / 4, 1039 (1997).

Outputs (to --out): montecarlo_hist.png, montecarlo_projection.png,
montecarlo_summary.csv, montecarlo_positives_*.csv
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
from collision_operators import m_p_g, m_B_g, m_e_g, e_esu, Z_B   # noqa: E402
from power_balance import E_rest_keV  # noqa: E402
sys.stdout = _real

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402
from scipy.stats import qmc           # noqa: E402

keV_to_erg = cs.keV_to_erg
barn_cm2 = cs.barn_cm2
Q_erg = cs.Q_pB11_keV * keV_to_erg
LNL = 17.0
E_CM_PEAK = 600.0
E_TAIL_LAB = E_CM_PEAK * (m_p_g + m_B_g) / m_B_g          # keV (lab)
E_TAIL_LAB_ERG = E_TAIL_LAB * keV_to_erg
V_TAIL = np.sqrt(2 * E_TAIL_LAB_ERG / m_p_g)
DISP = {'wang': 'Wang', 'TB': 'TB'}  # publication display names
SYM = {'eta_ch': '$\\eta_\\mathrm{ch}$', 'r_ash': '$n_\\alpha/n_e$',
       'tau': '$T_e/T_i$', 'f_tail': '$f_\\mathrm{tail}$',
       'eta_drive': '$\\eta_\\mathrm{drive}$'}
# slowing-down SI constants (Goldston-Rutherford, alpha; A/Z^2 same for proton)
_M_E_KG, _M_A_KG = 9.1093837015e-31, 6.6446573357e-27
_E_C, _EPS0 = 1.602176634e-19, 8.8541878128e-12
_TAU_PREF = (3 * np.sqrt(2 * np.pi) / (16 * np.pi)) * _M_A_KG * (4 * np.pi * _EPS0)**2 \
    / (np.sqrt(_M_E_KG) * (2**2) * _E_C**4 * LNL)


def tau_s_vec(n_e_cm3, Te_keV):
    """Vectorized slowing-down time (s); proton tail == alpha (same A/Z^2)."""
    T_e_J = Te_keV * 1e3 * _E_C
    return _TAU_PREF * T_e_J**1.5 / (n_e_cm3 * 1e6)


def build_interpolators():
    """Precompute <sigma v>(Ti) (per cross section) and R(Te); SELF-CHECK them."""
    Ti_grid = np.linspace(150.0, 600.0, 901)        # 0.5 keV spacing
    Te_grid = np.linspace(40.0, 600.0, 561)         # 1.0 keV spacing
    sv = {}
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        sv[xs] = cs.sigma_v_TB_numerical(Ti_grid)
    R_grid = np.array([pb.relativistic_R_factor(T) for T in Te_grid])
    sig600 = {}
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        sig600[xs] = cs.sigma_fusion(E_CM_PEAK)[0] * barn_cm2

    # ---- SELF-CHECK ----
    print("=" * 74)
    print("INTERPOLATION SELF-CHECK (must be < 0.1% vs full integrator)")
    print("=" * 74)
    ok = True
    for xs in ('wang', 'TB'):
        cs.CROSS_SECTION = xs
        for T in (175.3, 237.7, 313.1, 411.9, 533.3):
            full = cs.sigma_v_TB_numerical(T)[0]
            ip = float(np.interp(T, Ti_grid, sv[xs]))
            err = abs(ip - full) / full * 100
            ok &= err < 0.1
            print(f"  <sv> {xs:<4} Ti={T:6.1f}: full={full:.5e} interp={ip:.5e} "
                  f"err={err:.4f}%  {'OK' if err < 0.1 else 'FAIL'}")
    for T in (58.0, 131.0, 244.0, 377.0, 521.0):
        full = pb.relativistic_R_factor(T)
        ip = float(np.interp(T, Te_grid, R_grid))
        err = abs(ip - full) / full * 100
        ok &= err < 0.1
        print(f"  R(Te) Te={T:6.1f}: full={full:.5f} interp={ip:.5f} "
              f"err={err:.4f}%  {'OK' if err < 0.1 else 'FAIL'}")
    if not ok:
        print("\nSELF-CHECK FAILED — interpolation unreliable, ABORTING scan.")
        sys.exit(2)
    print("SELF-CHECK PASSED.\n")
    return Ti_grid, sv, Te_grid, R_grid, sig600


def brems(n_e, Te, Z_eff):
    P_cl = 5.34e-31 * n_e**2 * Z_eff * np.sqrt(Te)
    x = Te / E_rest_keV
    f_ei = 1 + 1.78 * x**1.34
    f_ee = 2.12 * x * (1 + 1.1 * x + x**2 - 1.25 * x**2.5) / Z_eff
    return P_cl * (f_ei + f_ee)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log2-samples', type=int, default=21)   # 2^21 ~ 2.1M
    ap.add_argument('--auto-escalate', action='store_true', default=True)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(_THIS, '..', 'figures')))
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Ti_grid, SV, Te_grid, R_grid, SIG600 = build_interpolators()

    # lever bounds
    lo = np.array([150., 0.30, 0.05, 0.001, 0.0, 0.10, 0.30, 0.0, 0.0])
    hi = np.array([600., 1.00, 0.30, 0.05, 0.5, 0.60, 0.80, 0.02, 0.30])
    names = ['Ti', 'tau', 'f_B', 'r_ash', 'eta_ch', 'f_tail', 'eta_drive',
             'f_imp', 'xi_sync']

    def run_scan(m):
        sob = qmc.Sobol(d=9, scramble=True, seed=args.seed)
        u = sob.random_base2(m)                     # (2^m, 9) in [0,1)
        X = lo + u * (hi - lo)
        Ti, tau, f_B, r_ash, eta_ch, f_tail, eta_drive, f_imp, xi = X.T
        Te = tau * Ti
        n_i = 1e14
        n_p = (1 - f_B) * n_i
        n_B = f_B * n_i
        n_e0 = n_p + Z_B * n_B
        n_alpha = r_ash * n_e0
        R = np.interp(Te, Te_grid, R_grid)
        frac_e = np.minimum(0.15 * (150.0 / np.maximum(Te, 50.0))**0.5, 0.5)

        results = {}
        for xs in ('wang', 'TB'):
            sv = np.interp(Ti, Ti_grid, SV[xs])
            P_fus_th = n_p * n_B * sv * Q_erg / 1e7
            P_ch = eta_ch * P_fus_th
            P_drive = (1.0 / eta_drive - 1.0) * P_ch
            P_relax_tail = f_tail * P_ch
            P_alpha_e = (1 - eta_ch) * P_fus_th * frac_e
            for Zimp in (6, 26):
                n_e = (n_p + Z_B * n_B + 2 * n_alpha) / (1 - f_imp)
                Zeff = (n_p + n_B * Z_B**2 + n_alpha * 4 + f_imp * n_e * Zimp) / n_e
                tau_s = tau_s_vec(n_e, Te)
                n_tail = (f_tail * P_ch * 1e7) * tau_s / E_TAIL_LAB_ERG
                dP_fus = (n_tail * n_B * SIG600[xs] * V_TAIL) * Q_erg / 1e7
                P_fus = P_fus_th + dP_fus
                P_brems = brems(n_e, Te, Zeff)
                P_sync = xi * P_brems
                # collision e-i transfer (vectorized P_ion_electron_transfer)
                base = (4 * np.sqrt(2 * np.pi) / 3) * n_e * e_esu**4 * LNL \
                    / (m_e_g**0.5 * (Te * keV_to_erg)**1.5)
                nu_pe = (m_e_g / m_p_g) * base * 1**2
                nu_Be = (m_e_g / m_B_g) * base * Z_B**2
                P_ie = (1.5 * n_p * nu_pe * (Ti - Te) * keV_to_erg * R
                        + 1.5 * n_B * nu_Be * (Ti - Te) * keV_to_erg * R) / 1e7
                relax_i = np.maximum(0.0, P_alpha_e + P_ie - P_brems)
                common = P_fus - P_brems - P_sync - P_drive - P_relax_tail
                net_i = common - relax_i
                net_ii = common - P_ie
                for vi, net in (('i', net_i), ('ii', net_ii)):
                    results[(xs, Zimp, vi)] = net / P_brems
        return X, results

    m = args.log2_samples
    X, results = run_scan(m)
    # auto-escalate if pure-Rider Wang has zero positives
    wang_ii_pos = sum(int(np.any(results[('wang', z, 'ii')] > 0)) for z in (6, 26))
    if args.auto_escalate and wang_ii_pos == 0 and m < 22:
        print(f"[escalate] 0 positive (Wang, pure-Rider) at 2^{m}; "
              f"re-running at 2^22 (~4.2M)...\n")
        m = 22
        X, results = run_scan(m)

    Ti, tau, f_B, r_ash, eta_ch, f_tail, eta_drive, f_imp, xi = X.T
    N = X.shape[0]

    # demonstrated-physics box (vs theoretical ceiling)
    DEMO = dict(eta_ch=lambda v: v <= 0.20, tau=lambda v: v >= 0.40,
                r_ash=lambda v: v >= 0.01)

    # ---- summary ----
    rows = []
    print("=" * 88)
    print(f"COMBINATORIAL SCAN  N = {N:,} Sobol samples")
    print("=" * 88)
    print(f"{'xsec':<5}{'Zimp':<6}{'relax':<7}{'N_pos':>10}{'per_1e6':>10}"
          f"{'max_ratio':>11}{'demo_pos':>10}")
    pos_store = {}
    for xs in ('wang', 'TB'):
        for Zimp in (6, 26):
            for vi in ('i', 'ii'):
                ratio = results[(xs, Zimp, vi)]
                pos = ratio > 0
                npos = int(pos.sum())
                per1e6 = npos / N * 1e6
                mx = float(ratio.max())
                demo_mask = pos & DEMO['eta_ch'](eta_ch) & DEMO['tau'](tau) \
                    & DEMO['r_ash'](r_ash)
                ndemo = int(demo_mask.sum())
                rows.append([xs, Zimp, vi, npos, f'{per1e6:.3f}', f'{mx:.4f}',
                             ndemo])
                print(f"{xs:<5}{Zimp:<6}{vi:<7}{npos:>10,}{per1e6:>10.3f}"
                      f"{mx:>11.4f}{ndemo:>10,}")
                if npos > 0:
                    pos_store[(xs, Zimp, vi)] = pos

    # ---- positives parameter dump + clustering (cap rows) ----
    sum_csv = os.path.join(args.out, 'montecarlo_summary.csv')
    with open(sum_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cross_section', 'Z_imp', 'relax_variant', 'N_total',
                    'N_positive', 'positive_per_1e6', 'max_Pnet_Pbrems',
                    'N_positive_in_demonstrated_box'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], N, r[3], r[4], r[5], r[6]])

    for key, pos in pos_store.items():
        xs, Zimp, vi = key
        idx = np.where(pos)[0]
        if len(idx) > 50000:
            idx = idx[np.linspace(0, len(idx) - 1, 50000).astype(int)]
        fn = os.path.join(args.out,
                          f'montecarlo_positives_{xs}_Z{Zimp}_{vi}.csv')
        with open(fn, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(names + ['Pnet_Pbrems'])
            for j in idx:
                w.writerow([f'{X[j, k]:.4f}' for k in range(9)]
                           + [f'{results[key][j]:.4f}'])

    # ---- figure 1: histograms ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ic, xs in enumerate(('wang', 'TB')):
        for k, vi in enumerate(('i', 'ii')):
            ax = axes[ic][k]
            for Zimp, c in ((6, 'tab:green'), (26, 'tab:red')):
                rr = np.clip(results[(xs, Zimp, vi)], -2.0, 0.6)
                ax.hist(rr, bins=160, histtype='step', color=c, lw=1.8,
                        label=f'$Z_\\mathrm{{imp}}={Zimp}$', log=True)
            ax.axvline(0.0, color='k', ls='--', lw=1.5)
            ax.set_xlabel('$P_\\mathrm{net}/P_\\mathrm{brems}$')
            ax.set_ylabel('count (log)')
            ax.set_title(f'{DISP[xs]} — $P_\\mathrm{{relax}}$ variant ({vi})')
            ax.legend(fontsize=9)
    fig.suptitle('p-$^{11}$B alpha-channeling: net-power distribution over '
                 f'{N:,} Sobol samples (dashed = ignition threshold)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out1 = os.path.join(args.out, 'montecarlo_hist.png')
    fig.savefig(out1, dpi=300, bbox_inches='tight')
    fig.savefig(out1.replace('.png', '.pdf'), bbox_inches='tight')  # vector for submission
    plt.close(fig)

    # ---- figure 2: projection of positives (where they live) ----
    # prefer the strictest case with positives: Wang variant ii; else Wang i
    proj_key = None
    for cand in [('wang', 6, 'ii'), ('wang', 26, 'ii'),
                 ('wang', 6, 'i'), ('wang', 26, 'i'),
                 ('TB', 6, 'ii'), ('TB', 6, 'i')]:
        if cand in pos_store:
            proj_key = cand
            break
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    if proj_key is not None:
        pos = pos_store[proj_key]
        rr = results[proj_key][pos]
        planes = [('eta_ch', 'r_ash'), ('eta_ch', 'tau'), ('f_tail', 'eta_drive')]
        cols = {n: X[pos, i] for i, n in enumerate(names)}
        for ax, (px, py) in zip(axes2, planes):
            sc = ax.scatter(cols[px], cols[py], c=rr, s=4, cmap='viridis')
            ax.set_xlabel(SYM.get(px, px))
            ax.set_ylabel(SYM.get(py, py))
            plt.colorbar(sc, ax=ax, label='$P_\\mathrm{net}/P_\\mathrm{brems}$')
        fig2.suptitle(f'Positive-island samples: {DISP[proj_key[0]]}, $Z_\\mathrm{{imp}}$='
                      f'{proj_key[1]}, $P_\\mathrm{{relax}}$ ({proj_key[2]}) — '
                      f'{int(pos.sum()):,} of {N:,}', fontsize=12)
    else:
        for ax in axes2:
            ax.text(0.5, 0.5, 'NO positive samples\nin any case',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
        fig2.suptitle('No positive net-power island found in the defensible space',
                      fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = os.path.join(args.out, 'montecarlo_projection.png')
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    fig2.savefig(out2.replace('.png', '.pdf'), bbox_inches='tight')  # vector for submission
    plt.close(fig2)

    print("-" * 88)
    print(f"[OK] {out1}\n[OK] {out2}\n[OK] {sum_csv}")
    print("DEMO box = eta_ch<=0.20 AND tau>=0.40 AND r_ash>=0.01 "
          "(demonstrated, not theoretical-ceiling)")


if __name__ == '__main__':
    main()
