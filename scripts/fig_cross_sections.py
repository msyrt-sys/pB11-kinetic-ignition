#!/usr/bin/env python3
"""
fig_cross_sections.py — Fig. 6: p-11B cross section sigma(E) and Maxwellian
reactivity <sigma v>(T) for the three parameterizations (Tentori-Belloni 2023,
Wang 2026, Nevins-Swain 2000), all evaluated through the identical reactivity
integrator (cross_sections.sigma_v_TB_numerical).

Output: figures/cross_section_comparison.png
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_THIS, '..', 'src'))
sys.path.insert(0, _SRC)
import cross_sections as cs   # noqa: E402

import matplotlib            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.normpath(os.path.join(_THIS, '..', 'figures',
                                    'cross_section_comparison.png'))
COL = {'TB': 'tab:red', 'wang': 'tab:blue', 'NS': 'tab:green'}
LAB = {'TB': 'Tentori–Belloni 2023', 'wang': 'Wang 2026',
       'NS': 'Nevins–Swain 2000'}
SIG = {'TB': cs.sigma_TB, 'wang': cs.sigma_wang, 'NS': cs.sigma_NS}


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) sigma(E)
    E = np.logspace(np.log10(50), np.log10(3500), 800)
    for k in ('TB', 'wang', 'NS'):
        ax[0].loglog(E, SIG[k](E), color=COL[k], lw=2, label=LAB[k])
    ax[0].axvspan(200, 700, color='gray', alpha=0.12,
                  label='reactivity-weighted band (T~300 keV)')
    ax[0].set_xlabel('CM energy $E$ (keV)', fontsize=12)
    ax[0].set_ylabel('$\\sigma(E)$ (barn)', fontsize=12)
    ax[0].set_title('(a) fusion cross section', fontsize=12)
    ax[0].legend(fontsize=9, loc='lower right')
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].set_xlim(50, 3500)

    # (b) <sigma v>(T)
    T = np.linspace(100, 600, 60)
    sv = {}
    for k in ('TB', 'wang', 'NS'):
        cs.CROSS_SECTION = k
        sv[k] = cs.sigma_v_TB_numerical(T)
        ax[1].plot(T, sv[k] * 1e16, color=COL[k], lw=2.2, label=LAB[k])
    ax[1].axvline(300, color='k', ls=':', lw=1)
    ax[1].annotate('300 keV:\nTB 4.37, Wang 3.63, NS 3.52',
                   xy=(300, 3.6), xytext=(330, 1.9), fontsize=8.5,
                   arrowprops=dict(arrowstyle='->', alpha=0.6))
    ax[1].set_xlabel('Ion temperature $T$ (keV)', fontsize=12)
    ax[1].set_ylabel('$\\langle\\sigma v\\rangle$ ($10^{-16}$ cm³/s)', fontsize=12)
    ax[1].set_title('(b) Maxwellian reactivity (identical integrator)', fontsize=12)
    ax[1].legend(fontsize=9, loc='upper left')
    ax[1].grid(True, alpha=0.3)

    fig.suptitle('p-$^{11}$B cross section and reactivity: three parameterizations '
                 'on an identical footing', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    fig.savefig(OUT.replace('.png', '.pdf'), bbox_inches='tight')  # vector for submission
    plt.close(fig)
    print('[OK]', OUT)
    print('TB/Wang(300)=%.3f  TB/NS(300)=%.3f'
          % (sv['TB'][np.argmin(abs(T - 300))] / sv['wang'][np.argmin(abs(T - 300))],
             sv['TB'][np.argmin(abs(T - 300))] / sv['NS'][np.argmin(abs(T - 300))]))


if __name__ == '__main__':
    main()
