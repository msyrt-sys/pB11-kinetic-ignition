# Cross-section sensitivity and the fragility of net-power windows in proton–boron-11 fusion: a multi-regime kinetic study

**Mehmet Enes Süzek**

Department of Internal Medicine, Faculty of Medicine, Afyonkarahisar Health Sciences University, 03200 Afyonkarahisar, Türkiye
ORCID: 0009-0006-7825-6809 · zzenfal97@gmail.com

---

## Abstract

Aneutronic proton–boron-11 (p-11B) fusion has been argued, on the basis of updated cross-sections and kinetic effects, to admit a marginal thermonuclear ignition window. We test the robustness of this conclusion with a self-consistent, zero-dimensional kinetic power-balance model employing a Maxwell-preserving Fokker–Planck discretization, and we extend the analysis across confinement regimes. Integrating three modern cross-section parameterizations through an identical reactivity integrator, we find the thermal reactivity at 300 keV spans ~20% (Tentori–Belloni 4.4×10⁻¹⁶, Wang 3.6×10⁻¹⁶, Nevins–Swain 3.5×10⁻¹⁶ cm³/s), and that this spread alone determines whether the thermal fusion-to-bremsstrahlung ratio sits above (~1.20) or at (~1.01) threshold under an identical self-consistent electron temperature (T_e/T_i ~ 0.42). The surviving net gain in the kinetic peak rests entirely on the alpha-driven R-matrix enhancement; with the Coulomb (Putvinski) enhancement alone it is sub-ignition. We further show the two principal kinetic uncertainties—steady-state alpha density and the R-matrix factor—are mathematically degenerate, so ignition is governed by a single effective parameter. Extending to a four-term net-power budget, we scan 4×10⁶ parameter combinations and find no positive net-power island under conservative (Rider) accounting; a time-dependent treatment shows the transient T_e<T_i window yields cycle-integrated gain ≪ 1; and a 1-D treatment shows spatial separation is dominated by transport losses. Across equilibrium, combinatorial, transient, and spatial regimes, conservatively-modeled p-11B ignition remains marginal to sub-threshold. Reproducible cross-section benchmarking and uncertainty propagation are prerequisites for any p-11B ignition claim.

**Keywords:** proton–boron-11 fusion; aneutronic fusion; thermonuclear ignition; bremsstrahlung; fusion reactivity; kinetic power balance; alpha channeling.

---

## I. Introduction

The aneutronic proton–boron-11 reaction, p + ¹¹B → 3α + 8.68 MeV, is attractive for fusion energy because it releases its energy almost entirely in charged particles and produces no primary neutrons. Its drawback is equally well known: the reactivity peaks near 600 keV and the bremsstrahlung radiated by the high-Z, high-temperature electrons is comparable to the fusion power, so that thermal p-11B operates at best near the boundary of net energy production. Quantitative zero-dimensional power balances [1] have placed the peak fusion-to-bremsstrahlung ratio only a few percent above unity, with onerous confinement requirements.

Two developments have renewed interest in a possible ignition window. First, re-evaluations of the fusion cross section [2, 3] revise the reactivity relative to the long-standing Nevins–Swain parametrization [4]. Second, kinetic effects—suprathermal proton tails sustained by slowing-down alpha particles, and nuclear (R-matrix) enhancement of large-angle alpha–proton elastic scattering [5]—have been proposed to raise the fusion power above its Maxwellian value. Because the underlying power balance is near unity, the conclusion (ignition window present or absent) is sensitive to each of these inputs and to their assumed magnitudes.

This sensitivity makes p-11B a problem in which *verification and uncertainty propagation are themselves the scientific content*. Recent Physics of Plasmas studies have emphasized exactly this point for advanced-fuel power balances [11]. The present work asks a single, falsifiable question: **across the confinement regimes available to a 0-D/1-D treatment, and under conservative accounting of every loss channel, does a positive net-power window for p-11B exist, and on which modeling assumptions does the answer depend?**

We answer this by (i) implementing three cross-section parameterizations behind one reactivity integrator (Sec. II.A–B), (ii) propagating the kinetic uncertainties through an explicit, self-consistent power balance (Sec. II.C–E, III.A–C), and (iii) extending the budget from the equilibrium baseline across combinatorial, time-dependent, and spatial regimes with the recirculating and transport costs retained (Sec. III.D). The result is a consistent, reproducible map of where p-11B ignition sits relative to threshold.

---

## II. Methods

### II.A Three cross sections, one integrator

We evaluate the Maxwellian reactivity

⟨σv⟩(T) = √(8/πμ) · (k_BT)^(−3/2) · ∫₀^∞ E σ(E) exp(−E/k_BT) dE,

with E the center-of-mass (relative) energy and μ the p-11B reduced mass (μc² = 859.5 MeV), for three astrophysical-S-factor parameterizations evaluated through the **same** integrator and the same Gamow form σ(E) = [S(E)/E] exp(−√(E_G/E)), E_G = 22.589 MeV:

- **Tentori–Belloni 2023 (TB)** [2] — Table 1 "this work" column, three energy segments with bare-amplitude Breit–Wigner resonances;
- **Wang et al. 2026 (Wang)** [3] — Table 1, five resonances, the 148 keV narrow resonance retained;
- **Nevins–Swain 2000 (NS)** [4] — the "Nevins and Swain" reference column of TB Table 1, evaluated identically.

All energies are kept in the center-of-mass frame throughout the reactivity integral; the laboratory↔CM conversion (E_lab = E_cm·12/11 for a proton on a boron nucleus) enters only in the kinetic (lab-frame distribution) calculation of Sec. II.D. Integration limits and the resonance line-shape convention are stated in the released source.

> **Footnote (NS reactivity).** Our NS reactivity values are obtained by direct S-factor integration through the common integrator. The widely-cited NS-2000 *analytic* reactivity fit (Bosch–Hale/Peres form) is stated valid only to ~500 keV and plateaus beyond [4]; evaluating it gives ⟨σv⟩ = 3.385×10⁻¹⁶ cm³/s at 300 keV, so that our NS S-factor integral lies **+3.9% (300 keV), +9.2% (500 keV), and +11.4% (600 keV)** above the analytic fit, the offset growing as the fit reaches its stated validity edge. Computing all three parameterizations through one faithful integrator—rather than mixing an analytic fit with numerical integrals—is a deliberate choice for internal consistency, not an error; the cross-section coefficients were verified three independent ways against the published TB Table 1, and the analytic-fit offset is reproduced by the released code.

### II.B Reactivity benchmark

Table 1 reports ⟨σv⟩(T) for the three parameterizations from 100 to 600 keV. At 300 keV, **TB = 4.37×10⁻¹⁶, Wang = 3.63×10⁻¹⁶, NS = 3.52×10⁻¹⁶ cm³/s**. The TB/Wang ratio is nearly flat at ~1.20; the TB/NS ratio rises from 1.24 (300 keV) to 1.45 (600 keV). The TB curve agrees with the original Tentori–Belloni tabulation to within 0.4%, and the low-temperature NS values are consistent with independent recent evaluations [12], and the parameterized cross sections reproduce the measured ¹¹B(p,α) S-factor [10]. Figure 1 shows σ(E) and ⟨σv⟩(T) for all three. The ~20% spread in the reactivity is the single largest modeling uncertainty entering the power balance and is propagated explicitly below.

### II.C Maxwell-preserving Fokker–Planck operator

The proton distribution is evolved with an isotropic, one-dimensional (speed) Fokker–Planck operator using the Trubnikov diffusion and friction coefficients for proton–proton, proton–boron, and proton–electron collisions, plus a non-Maxwellian alpha contribution (Sec. II.D). The diffusion/friction functionals are evaluated **directly at the cell half-points** rather than by linear averaging; this preserves the Maxwell–Boltzmann steady state to a relative residual ~10⁻¹¹ (versus ~10⁻² for the linear-average scheme), verified by the detailed-balance test L[f_M] ≈ 0.

We stress the scope of this figure of merit: exact preservation of a Maxwellian is a *necessary* property of a kinetic solver but does not by itself validate the non-Maxwellian tail that carries the kinetic enhancement. The tail accuracy is assessed separately through the enhancement decomposition of Sec. III.B. The full velocity-space discretization—grid, half-point coefficient evaluation, and boundary conditions—is documented with the released source (Sec. Data availability).

### II.D R-matrix kinetic enhancement as a sensitivity input

The alpha-driven enhancement of the proton tail is computed from the slowing-down alpha distribution coupling to protons via Trubnikov (Coulomb) diffusion and, optionally, the Belloni 2021 nuclear enhancement [5] of large-angle α–p elastic scattering. The latter is obtained from a three-level (S-, P-, D-wave) ⁵Li R-matrix phase-shift evaluation, returning the energy-dependent factor F(v_p) = ⟨dσ/dE_p⟩_total / ⟨dσ/dE_p⟩_Rutherford. Because the R-matrix scattering cross section agrees with reference evaluations only to within a factor of order two, **F enters the analysis as a scanned sensitivity parameter, not a fixed number**: the nominal R-matrix factor (the unscaled Belloni evaluation) is multiplied by a dimensionless scale F_scale ∈ [0.5, 2] spanning the factor-two scattering uncertainty against SigmaCalc 2.0, with the nominal operating point at (n_α/n_e, F_scale) = (0.05, 1.0) and the conservative point at (0.01, 0.5) (Sec. III.B–C).

### II.E Power balance

Local power densities (W/cm³) are computed explicitly:

- **Fusion:** P_F = n_p n_B ⟨σv⟩(T_i) Q, Q = 8.68 MeV (all charged).
- **Bremsstrahlung:** the NRL classical form with the Svensson relativistic correction, P_B = 5.34×10⁻³¹ · n_e² · Z_eff · √(T_e) · g(x), x = T_e/m_ec². The coefficient is the standard NRL classical value [8] evaluated in **mixed practical units** (n_e in cm⁻³, T_e in keV, P_B in W/cm³); it is equivalent to 1.69×10⁻³² with T_e in eV, and to 5.34×10⁻³⁷ in SI units (n_e in m⁻³, P_B in W/m³). The correction g(x) contains the e–i and e–e relativistic terms (Svensson form) and was cross-checked against the recent independent analytical bremsstrahlung fit of Xie [9], which spans the e–i and e–e contributions to <1% over the relevant temperature range. (We adopt the NRL classical coefficient; the Wang 2026 power balance uses a value ~3% lower, a Gaunt/Born convention difference — at the near-threshold thermal margin a further small modeling sensitivity, Sec. III.D.) Z_eff and n_e follow a single ash-inclusive definition n_e = n_p + Z_B n_B + Z_α n_α, Z_eff = Σ n_i Z_i² / n_e.
- **Ion–electron exchange:** P_ie from the Spitzer/Trubnikov energy-equilibration rate with the relativistic correction R(x); this matches the NRL Plasma Formulary expression.
- **Self-consistent T_e:** determined by the electron power balance P_αe + P_ie = P_B, yielding T_e/T_i ≈ 0.42 over the relevant range (Sec. III.A), i.e. a naturally decoupled hot-ion mode.
- **Net-power budget (Sec. III.D):** the multi-regime boundary study charges every loss channel, P_net = P_fus − P_brems − P_drive − P_relax. The wave-drive cost of channeling a power P_ch = η_ch P_α (η_ch ≤ 0.5 the channeling efficiency, η_drive the wave-drive efficiency, P_α the alpha power) is P_drive = (1/η_drive − 1) P_ch. The recirculation splits as P_relax = P_{relax,tail} + P_{relax,dec}: continuous re-driving of the wave-sustained suprathermal tail, P_{relax,tail} = f_tail P_ch (the tail thermalizes on the slowing-down time), plus a decoupling cost P_{relax,dec} bracketed by two variants — (i) no-double-count, P_{relax,dec} = max(0, P_αe + P_ie − P_B), the electron-power excess over bremsstrahlung that vanishes at the self-consistent T_e; and (ii) strict Rider, P_{relax,dec} = P_ie, the full ion→electron collisional transfer treated as recirculating load. The combinatorial scan additionally charges a synchrotron loss P_sync = ξ P_B and an impurity contribution to n_e and Z_eff.

> **Footnote (Coulomb logarithm).** The Wang thermal peak is sensitive to the electron–ion Coulomb logarithm: P_F/P_B(peak) = 1.01, 1.05, 1.12, 1.18 for lnΛ = 17, 15, 12, 10. At the standard value lnΛ ≈ 17 the result is at threshold; the spread is a further modeling sensitivity of the near-threshold conclusion.

The principal results were reproduced by an independent from-scratch reimplementation and cross-checked against the published external references cited above; the cross-check is qualitative (the implementations concur on the quantities tabulated here), and the data-availability statement gives the reproducible pipeline.

---

## III. Results

### III.A Thermal baseline and cross-section sensitivity

With a single self-consistent electron temperature, the **thermal** (Maxwellian) fusion-to-bremsstrahlung ratio peaks at

- **TB: P_F/P_B = 1.20 at T_i = 364 keV**,
- **Wang: 1.01 at 381 keV** (exactly at threshold),
- **NS: 0.96 at 316 keV** (sub-threshold).

The optimum boron fraction is f_B ≈ 0.13 for both modern cross sections, and re-optimizing f_B changes the peak by <1%. Thus the cross-section choice alone moves the thermal result from comfortably igniting (TB) to threshold (Wang) to sub-threshold (NS)—the ~20% reactivity spread of Table 1 maps directly onto the sign of the ignition margin.

### III.B Kinetic enhancement, decomposed

The kinetic enhancement of the fusion power, enh ≡ P_F[f_kinetic]/P_F[Maxwell], decomposes as shown in Table 2 (Wang, T_i = 180 keV).

The kinetic-peak ratio is **P_F/P_B = 1.26 (Wang)** and **1.50 (TB)**. Removing the R-matrix factor and retaining only the Coulomb (Putvinski) enhancement gives **P_F/P_B = 0.98 (Wang) — sub-ignition**. The surviving net gain therefore rests entirely on the R-matrix increment, the input carrying the largest (factor-two) uncertainty.

### III.C Degeneracy of the kinetic uncertainties

The alpha→proton diffusion scales as the product of the steady-state alpha density and the R-matrix factor, D_α ∝ (n_α/n_e)·F_scale. Consequently the kinetic enhancement depends only on the single combined parameter **p = (n_α/n_e)·F_scale**; we verify this collapse to machine precision (spread = 0 across off-diagonal pairs, with the R-matrix scale applied live). The two principal kinetic uncertainties are thus mathematically degenerate, and the ignition boundary in the (n_α, F_scale) plane is a hyperbola p = p*. We find **p*(Wang) = 0.013** and **p*(TB) = 0** (TB ignites across the box). Figure 2 shows the resulting 2-D sensitivity map; within the conservative band (channeling efficiency ≤ 0.5) the Wang result is sub-ignition over a finite region and razor-thin elsewhere. This is an *identifiability* statement: a single effective parameter, not two, governs the kinetic margin.

### III.D Net-power across regimes: a boundary study

We now retain *every* loss channel and ask whether a positive net-power window opens in any regime. The budget is P_net = P_fus − P_brems − P_drive − P_relax, where P_drive is the wave-drive cost of any channeling and P_relax is the recirculating cost of maintaining the non-equilibrium (the e–i recirculation à la Rider, and the suprathermal-tail maintenance). Two accounting variants bracket the recirculation: a no-double-count variant (electron-power excess over bremsstrahlung) and a strict (Rider) variant (full e–i transfer).

**Combinatorial (steady-state).** A Sobol scan of **4.19×10⁶** parameter combinations over seven simultaneously-free levers (ion temperature, T_e/T_i, boron fraction, alpha density, channeling efficiency ≤ 0.5, tail fraction, drive efficiency, plus impurity and synchrotron losses) finds, under the strict (Rider) accounting and the modern Wang cross section, **no positive net-power sample** (best case 14% below threshold). Under the lenient accounting only 2 of 4.19×10⁶ samples are positive, and both require *every* optimistic lever simultaneously pinned at its ceiling (channeling efficiency ≈ 0.49, forced electron cooling below the self-consistent value, near-zero synchrotron, aggressive ash removal); neither lies in the demonstrated-physics sub-box. Two independent accounting approaches bracket the conservative margin at **−14% to −76%**, both negative. Figure 3 shows the net-power distribution and the parameter projection of the rare positive samples.

**Time-dependent (transient).** A two-temperature 0-D model evolved with a stiff integrator (instantaneous hot-ion spark, finite confinement time τ_E) tests the transient T_e<T_i window. The instantaneous P_fus/P_brems reaches ~2.7 while the electrons are cold, but the **cycle-integrated gain** G = E_fus/(E_spark + E_brems + E_trans) is ~0.02 at the achievable τ_E (~1 s) and **remains below 0.4 even at τ_E = 100 s** (the perfect-confinement limit)—i.e. not even a confinement artifact. Crediting all alpha power to the ions (the most ion-favorable accounting) raises this only to G ≲ 0.36. The hot-ion spark energy exceeds the fusion produced before the plasma cools by a factor of order 50. Figure 4 shows representative trajectories and the G-map.

**Spatial (1-D radial).** A cylindrical steady-state model with prescribed hot-core/cold-edge profiles, charging the conductive transport loss, tests spatial separation of fusion from radiation/ash. At realistic heat diffusivities (Bohm and gyro-Bohm, B ∈ [5,20] T) the boundary heat conduction exceeds the volume-integrated fusion by **20× (gyro-Bohm) to ~10⁴× (Bohm)**; even in the χ→0 (perfect-insulation) limit the decoupled profile is sub-marginal, because concentrating fusion in a small hot core reduces the volume-integrated reactivity more than it saves edge bremsstrahlung. Spatial separation fails on both counts. Figure 5 shows the radial profiles and the net-power maps.

Table 3 consolidates the four regimes. Five distinct modeling sensitivities are reported transparently and propagate into the margins: the NS analytic-vs-integral offset (Sec. II.A), the ~3% bremsstrahlung-coefficient convention (Sec. II.E), the combinatorial accounting bracket (−14% to −76%), the pulsed alpha-deposition accounting (G ≤ 0.36), and the Coulomb logarithm (Sec. II.E).

---

## IV. Discussion

The picture that emerges is internally consistent and consistent with the literature. The thermal margin with the modern Wang cross section (~1.01) reproduces the few-percent margin and onerous confinement requirement of the original Putvinski analysis [1]; the present study is a *confirmation within uncertainty*, not a contradiction. The equilibrium, combinatorial, transient, and spatial regimes provide four independent realizations of the recirculating-power limit on non-equilibrium fusion systems [7]: maintaining the non-thermal feature that helps (a suprathermal tail, a two-temperature state, a spatial gradient) costs more than it returns once the maintenance cost is charged.

The two most discussed enhancement mechanisms enter as follows. The alpha-channeling/ash-demixing schemes [6], and subsequent spatial-separation proposals, can in principle reduce bremsstrahlung and feed the reactive tail, but in the present budget their drive and recirculation costs offset the gain at demonstrated efficiencies; the positive island, where it exists at all, requires channeling efficiencies and ash-removal rates well above demonstrated values and is therefore an unproven precondition rather than a result. The R-matrix elastic enhancement is the single lever that converts the Wang kinetic peak from sub-ignition to ~1.26; given its factor-two uncertainty, a quantitative ignition claim cannot rest on it without an independent calibration.

We deliberately avoid characterizing earlier optimistic estimates as artifacts; the differences are quantitatively attributable to specific, individually-defensible modeling choices (cross-section parameterization, enhancement magnitude, loss accounting), and the contribution of this work is to make those choices explicit and to propagate their uncertainty. The 0-D/1-D scope is also a limitation: a positive island, were one to survive a full transport calculation, would not be excluded by the present treatment—but the consistent negativity across four independent regimes, each biased toward the optimistic side where assumptions were required, places the burden of proof on any future positive claim.

---

## V. Conclusion

Across equilibrium (thermal and kinetic), combinatorial (4.19×10⁶-sample), time-dependent (transient), and spatial (1-D) regimes, conservatively-modeled proton–boron-11 ignition is marginal to sub-threshold with the modern Wang cross section, and the surviving optimistic strand is fragile to individual modeling choices: the cross-section parameterization alone flips the thermal margin across threshold; the kinetic net gain rests entirely on a factor-two-uncertain R-matrix factor; and no defensible parameter combination yields a positive net-power island under conservative accounting. The two principal kinetic uncertainties are degenerate, so a single effective parameter controls the margin. Reproducible cross-section benchmarking and explicit uncertainty propagation are, on this evidence, prerequisites for any p-11B ignition claim. The framework, all cross-section coefficients, and every figure are released for reuse.

---

## Data availability

All numerical results are reproduced by the released code (branch `faithful-cross-section`) with a fixed environment (numpy 2.x, scipy 1.x, matplotlib 3.x). Each figure and table is generated by a named script with its CSV output:

- Table 1, Figure 1 — `scripts/fig_cross_sections.py` (cross sections + reactivity); a ⟨σv⟩ benchmark gate verifies the integrator to <0.15%.
- Figure 2 — `scripts/sensitivity_analysis_2d.py` (+ `sensitivity_2d_grid.csv`, `sensitivity_2d_summary.csv`).
- Figure 3 — `scripts/alpha_channeling_montecarlo.py` (+ `montecarlo_*.csv`); includes a mandatory interpolation self-check (<0.1%).
- Figure 4 — `scripts/pulsed_0d_model.py` (+ `pulsed_*.csv`); interpolation self-check (<0.1%).
- Figure 5 — `scripts/radial_1d_model.py` (+ `radial_*.csv`); interpolation self-check (<0.1%) and a uniform-profile → 0-D reduction self-check (<1%).

**Independent verification.** The principal results were reproduced by an independent from-scratch reimplementation, and anchored to published external evaluations (Tentori–Belloni 2023 cross section to 0.4%; Nevins–Swain low-temperature reactivity; the relativistic bremsstrahlung terms [9]; and the NRL ion–electron exchange rate). The cross-section coefficients were validated three independent ways against the published Table 1.

---

## References

1. S. V. Putvinski, D. D. Ryutov, P. N. Yushmanov, *Nucl. Fusion* **59**, 076018 (2019).
2. A. Tentori, F. Belloni, *Nucl. Fusion* **63**, 086001 (2023).
3. H.-Y. Wang, Y.-Q. Li, Q. Wu, Z.-F. Cui, *Revisiting p-11B Fusion: Updated Cross-sections, Reactivity, and Energy Balance*, arXiv:2601.00241 (2026).
4. W. M. Nevins, R. Swain, *Nucl. Fusion* **40**, 865 (2000).
5. F. Belloni, *Plasma Phys. Control. Fusion* **63**, 055020 (2021).
6. I. E. Ochs, E. J. Kolmes, M. E. Mlodik, T. Rubin, N. J. Fisch, *Phys. Rev. E* **106**, 055215 (2022), arXiv:2210.08076.
7. T. H. Rider, *Phys. Plasmas* **2**, 1853 (1995); **4**, 1039 (1997).
8. J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (Washington, DC) (ion–electron energy equilibration rate; Coulomb logarithm; classical bremsstrahlung coefficient).
9. R. Svensson, *Astrophys. J.* **258**, 335 (1982) (relativistic bremsstrahlung correction); H. Xie, *Plasma Phys. Control. Fusion* **66**, 125005 (2024), DOI 10.1088/1361-6587/ad877f, arXiv:2404.11540 (recent analytical e–i/e–e bremsstrahlung fitting, used as an independent cross-check).
10. M. Stave et al., *Phys. Lett. B* **696**, 26 (2011).
11. S. D. Baalrud, *Phys. Plasmas* **32**, 102709 (2025), DOI 10.1063/5.0292235; I. E. Ochs et al., *Phys. Plasmas* **33**, 012703 (2026) (bremsstrahlung constraints on p-11B IFE); I. Morozov, T. A. Mehlhorn, et al., *Phys. Plasmas* **33**, 042705 (2026), DOI 10.1063/5.0322446.
12. *Evaluation of the Lawson criterion for aneutronic proton–boron-11 fusion*, Frontiers Nucl. Eng. (2026), doi:10.3389/fnuen.2026.1714531.

---

## Tables

### Table 1 — Maxwellian reactivity ⟨σv⟩(T) (10⁻¹⁶ cm³/s), identical integrator

| T (keV) | TB 2023 | Wang 2026 | NS 2000 | TB/Wang | TB/NS |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.704 | 0.628 | 0.630 | 1.121 | 1.117 |
| 200 | 2.948 | 2.476 | 2.471 | 1.190 | 1.193 |
| 300 | 4.374 | 3.628 | 3.516 | 1.206 | 1.244 |
| 400 | 5.132 | 4.274 | 3.941 | 1.201 | 1.302 |
| 500 | 5.604 | 4.724 | 4.084 | 1.186 | 1.372 |
| 600 | 5.957 | 5.109 | 4.105 | 1.166 | 1.451 |

(CM-frame energy; E_G = 22.589 MeV; bare-amplitude Breit–Wigner S-factor; integration to the upper validity bound of each parameterization. Full 100–600 keV table at 50-keV resolution in the released CSV.)

### Table 2 — Kinetic enhancement decomposition (Wang, T_i = 180 keV)

| stage | enh | interpretation |
|---|---:|---|
| thermal FP (no alpha) | 0.83 | burnout depletes the tail |
| + Putvinski Coulomb alpha | 1.08 | +8% (consistent with the literature ~10%) |
| + Belloni R-matrix | 1.64 | ×1.5 increment from the nuclear factor |

(enh ≡ P_F[f_kinetic]/P_F[Maxwell]; the thermal Fokker–Planck steady state, then the additive Coulomb (Putvinski) alpha drive and the Belloni R-matrix nuclear factor.)

### Table 3 — Multi-regime net-power summary (Wang cross section, conservative accounting)

| Regime | Quantity | Result | Threshold |
|---|---|---|---|
| Thermal | peak P_F/P_B | 1.01 (TB 1.20; NS 0.96) | sits at threshold |
| Kinetic | peak P_F/P_B | 1.26 (0.98 without R-matrix) | needs R-matrix |
| Combinatorial | positive samples / 4.19×10⁶ (strict) | 0 (best −14%; bracket to −76%) | none |
| Transient | cycle gain G (achievable τ_E) | ~0.02 (<0.4 at τ_E=100 s; ≤0.36 α→ions) | ≪ 1 |
| Spatial 1-D | P_transport / P_fus (realistic χ) | 20× – 10⁴× | ≫ 1 |

---

## Figure captions

**Figure 1.** p-11B fusion cross section σ(E) (a) and Maxwellian reactivity ⟨σv⟩(T) (b) for the Tentori–Belloni 2023, Wang 2026, and Nevins–Swain 2000 parameterizations, all evaluated through the identical reactivity integrator. At 300 keV the reactivities are 4.37, 3.63, and 3.52×10⁻¹⁶ cm³/s.

**Figure 2.** Two-dimensional kinetic-uncertainty map. Because the enhancement depends only on p = (n_α/n_e)·F_scale, the ignition boundary is a hyperbola; the panel shows peak P_F/P_B over the (n_α, F_scale) plane for both cross sections, with the P_F/P_B = 1 contour, the defensible channeling-efficiency limit, and the Putvinski-only (no-R-matrix) reference.

**Figure 3.** Combinatorial net-power scan (4.19×10⁶ Sobol samples, seven free levers). Top: distribution of P_net/P_brems for the two cross sections and the two recirculation-accounting variants, with the ignition threshold marked; the strict-accounting Wang distribution lies entirely below threshold.

**Figure 4.** Time-dependent two-temperature model. Left/representative: T_i(t), T_e(t) and P_fus(t), P_brems(t) showing the transient T_e<T_i window (instantaneous P_fus/P_brems up to 2.7) and the cycle-integrated gain G ≪ 1. Right: G over the (T_{i,0}, τ_E) plane; G < 1 everywhere, including τ_E = 100 s.

**Figure 5.** One-dimensional radial model. Left: hot-core/cold-edge profiles and the conductive power crossing each radius versus the cumulative fusion power (transport exceeds fusion by ~10⁴× at Bohm, B=10 T). Right: P_net/∫P_brems and log₁₀(P_transport/∫P_fus) over the (T_{i,core}, B) plane for Bohm and gyro-Bohm diffusivity.
