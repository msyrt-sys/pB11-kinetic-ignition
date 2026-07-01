# Independent validation (from-scratch reimplementation)

This folder contains a **fully independent** re-derivation of the paper's central
results. It was written from scratch, shares **no code** with `../src/`, and exists
to test whether the conclusions survive a different implementation of the same
physics (addressing the reproducibility / independent-verification point).

- `p11b_independent_validation.py` — standalone reimplementation (numpy + scipy only).
- `p11b_results.json` — its archived output (regenerate with the command below).

## Why it is genuinely independent

It shares no source with the main repository and deliberately differs in its
numerical choices; only the **published S-factor coefficients** (NS-2000, TB-2023,
Wang-2026) are common, as they must be — both codes evaluate the same cross sections.

| aspect | main repo (`../src/`) | this validation |
|---|---|---|
| units | mixed practical (cm⁻³, keV) | SI throughout (m⁻³, J) |
| reactivity integral | repo's own integrator | `scipy.integrate.quad` |
| bremsstrahlung | NRL classical, 5.34×10⁻³¹ (≡ 5.34×10⁻³⁷ SI), Svensson g(x) | Wang/Rider form, 5.172×10⁻³⁷ SI (~3 % lower) |
| Coulomb logarithm | lnΛ = 17 | lnΛ = 12 |
| combinatorial sampling | Sobol QMC | uniform pseudo-random |
| pulsed model | repo stiff 0-D | own `scipy.solve_ivp` two-temperature ODE |

## What it confirms

Every number below is reproduced by running the script in a clean numpy/scipy
environment (verified against the shipped JSON to 0.0000 %).

**Reactivity anchors ⟨σv⟩ (cm³/s).** The two independent codes agree to **< 0.1 %**
at the 300 keV anchor:

| model | this validation (300 keV) | repo Table 1 (300 keV) |
|---|---|---|
| TB   | 4.374×10⁻¹⁶ | 4.374×10⁻¹⁶ |
| Wang | 3.628×10⁻¹⁶ | 3.628×10⁻¹⁶ |
| NS   | 3.516×10⁻¹⁶ | 3.516×10⁻¹⁶ |

TB and Wang also agree to < 0.01 % at 600 keV; the NS parameterization differs by
~1.4 % at 600 keV (an NS high-energy-tail sensitivity, not a coefficient error).
The NS S-factor **integral** runs +3.9 % (300 keV) growing to ~+13 % (600 keV) above
the NS-2000 *analytic* reactivity fit, independently reproducing the repo's
documented analytic-vs-integral offset.

**Thermal baseline (self-consistent Tₑ, f_B = 0.15), peak P_F/P_B:**

| model | this validation | repo (§III.A) |
|---|---|---|
| TB   | 1.22 @ 365 keV | 1.20 @ 364 keV |
| Wang | **1.01 @ 378 keV** | **1.01 @ 381 keV** |
| NS   | 0.96 @ 314 keV | 0.96 @ 316 keV |

Both codes independently place Wang exactly at threshold, TB above, NS below — the
paper's headline cross-section-sensitivity result.

**Combinatorial boundary study (4.2×10⁶ samples, Wang, strict/pure-Rider accounting):**
**0 positive net-power samples**, identical in conclusion to the repo's Sobol scan.

**Pulsed 0-D (transient Tₑ < Tᵢ):** cycle-integrated gain **G < 1 for every τ_E**,
including the perfect-confinement limit (τ_E = 10⁹ s): max G = 0.24 with no alpha
heating, and 0.73 even when **all** alpha power is credited to the ions. No breakeven
cycle exists.

**Spatial 1-D (radial):** boundary transport exceeds volume-integrated fusion by
**1× to ~10⁴×** across the radius/χ scan, and bremsstrahlung alone is **1.32× fusion**,
so the decoupled profile is sub-marginal even at χ → 0 (perfect insulation). Matches
the repo's spatial conclusion.

## Documented sensitivity (−14 % vs −76 %)

The two independent scans agree on the **qualitative** result (0 positive samples in
4.2×10⁶ under strict Rider accounting) but differ on the **best-case (least-negative)
margin**: the repo's Sobol scan reaches **−14 %** of threshold, whereas this
validation's best sample is **−76 %**. The difference traces to the different
bremsstrahlung coefficient (NRL vs Wang/Rider, ~3 %), the different lever ranges, and
the pure-Rider recirculation bookkeeping. The **no-positive-window conclusion is
robust** to this spread; the exact margin is implementation-dependent and is reported
here transparently rather than reconciled.

## Reproduce

```
python p11b_independent_validation.py > p11b_results.json
```

Requires numpy and scipy only. Runtime is dominated by the 4.2×10⁶-sample
combinatorial scan and the pulsed ODE integrations (order minutes).
