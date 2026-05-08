# Self-consistent kinetic ignition window for proton-boron-11 fusion

Source code accompanying the manuscript:

> **Self-consistent kinetic ignition window for proton-boron-11 fusion with the Tentori-Belloni cross section, R-matrix elastic scattering, and resilience to alpha-ash poisoning**
> 
> Mehmet Enes Süzek, Afyonkarahisar Health Sciences University (2026)
> 
> *Submitted to Physics of Plasmas*

This repository contains the complete Python implementation of the Fokker-Planck/Trubnikov framework used in the manuscript, together with the analysis scripts that produce all figures.

---

## Overview

The code implements a self-consistent kinetic ignition analysis for the aneutronic p-¹¹B fusion reaction, combining four updates to the standard Putvinski [Nucl. Fusion **59**, 076018 (2019)] treatment:

1. **Tentori-Belloni 2023 cross section** [Nucl. Fusion **63**, 086001] — ~50% larger reactivity than older parametrizations near *T* = 300 keV.
2. **Belloni 2021 R-matrix elastic scattering** [Plasma Phys. Control. Fusion **63**, 055020] — 3-level (S, P, D wave) phase-shift analysis with ⁵Li resonances.
3. **Stave-validated alpha-source spectrum** [Phys. Lett. B **696**, 26 (2011)] — two-component fit calibrated against experimental data.
4. **Maxwell-preserving Fokker-Planck operator** — half-point evaluation of diffusion/friction functionals achieves residual ~10⁻¹¹ (vs ~10⁻² for linear-average).

Principal results:
- Kinetic peak ratio **P_F/P_B = 1.50 at T_i = 183 keV** (vs Putvinski ~1.03)
- Optimum operating point: **T_i = 356 keV, f_B = 0.133, τ_E* = 102.7 s**
- Alpha-ash threshold: **~7.4% at T_i = 300 keV** without channeling (vs Ochs ~2%)

---

## Repository structure

```
.
├── src/                          # Core physics modules
│   ├── cross_sections.py         # Tentori-Belloni σ(E), σv(T)
│   ├── collision_operators.py    # Trubnikov D, F coefficients
│   ├── alpha_source.py           # α slowing-down distribution
│   ├── fp_solver.py              # Fokker-Planck integrator (CN + CC)
│   ├── power_balance.py          # P_F, P_B, ash poisoning, ignition
│   ├── belloni_full_implementation.py   # R-matrix elastic enhancement
│   ├── alpha_p_phaseshift.py     # 3-level R-matrix kernel
│   └── sigmacalc_alpha_p_data.py # SigmaCalc 2.0 validation data
├── scripts/
│   └── main_validation.py        # Reproduces all figures
├── figures/                      # PNG figures from manuscript
├── docs/
│   └── theory_notes.md           # Detailed derivations
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT license
└── README.md                     # This file
```

---

## Installation

Requires Python 3.10+ and standard scientific stack:

```bash
git clone https://github.com/<USERNAME>/pB11-kinetic-ignition.git
cd pB11-kinetic-ignition
pip install -r requirements.txt
```

Dependencies (also in `requirements.txt`):
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7

---

## Quick start

To reproduce all manuscript figures:

```bash
cd scripts
python main_validation.py
```

This runs the complete analysis pipeline (~30 minutes on a single CPU core) and produces 8 PNG figures in the `figures/` directory.

To run a single operating point:

```python
import sys
sys.path.insert(0, 'src')
from main_validation import fp_coupled_calculation
from power_balance import find_self_consistent_Te

n_p, n_B = 0.85e14, 0.15e14   # cm⁻³
T_i = 300.0                    # keV
T_e = find_self_consistent_Te(n_p, n_B, T_i)

result = fp_coupled_calculation(T_i, T_e, n_p, n_B, 
                                 include_belloni=True, 
                                 n_alpha_over_ne=0.05)
print(f"P_F (kinetic): {result['P_F_kinetic']:.4f} W/cm³")
print(f"Kinetic enhancement: {result['kinetic_enhancement']:.3f}")
```

---

## Reproducing manuscript figures

| Manuscript figure | Script function | Approximate runtime |
|:---|:---|:---:|
| Fig. 1 (TB cross section) | `plot_cross_section_comparison()` | < 10 s |
| Fig. 4 (Putvinski reproduction) | `plot_putvinski_fig4_reproduction()` | ~5 min |
| Fig. 5 (ignition map) | `plot_ignition_map()` | ~30 min |
| Fig. 3 (kinetic distortion) | `plot_kinetic_distortion()` | < 30 s |
| Fig. 8 (sensitivity) | `plot_sensitivity_analysis()` | ~10 min |
| Fig. 6 (ash 3-panel) | (to be added — see manuscript) | ~5 min |
| Fig. 7 (ash threshold 2D) | (to be added — see manuscript) | ~10 min |

---

## Validation tests

Several built-in validation tests confirm the implementation:

```bash
cd src
python fp_solver.py        # Maxwell preservation: residual ~10⁻¹¹
python collision_operators.py  # Trubnikov detailed balance: ~10⁻¹⁶
python cross_sections.py   # TB σ(E) reproduction
python belloni_full_implementation.py  # R-matrix vs SigmaCalc 2.0
```

---

## Citation

If you use this code in published work, please cite:

```bibtex
@article{Suzek2026_pB11,
  author  = {S{\"u}zek, Mehmet Enes},
  title   = {Self-consistent kinetic ignition window for proton-boron-11 
             fusion with the Tentori-Belloni cross section, R-matrix elastic 
             scattering, and resilience to alpha-ash poisoning},
  journal = {Phys. Plasmas},
  year    = {2026},
  note    = {Submitted}
}
```

A Zenodo DOI for this code repository will be added upon manuscript acceptance.

---

## License

This software is released under the MIT License — see `LICENSE`.

---

## Contact

**Mehmet Enes Süzek**
Department of Internal Medicine
Afyonkarahisar Health Sciences University, Türkiye
✉ zzenfal97@gmail.com
ORCID: [0009-0006-7825-6809](https://orcid.org/0009-0006-7825-6809)

---

## Acknowledgments

The author is grateful to A. Tentori and F. Belloni for making the parameters of their analytic cross section publicly available. This work was carried out independently of any institutional or external funding source.
