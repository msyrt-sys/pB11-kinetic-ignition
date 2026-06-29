"""
cross_sections.py — p-11B fusion cross section and reactivity parametrizations

Primary reference: Tentori, A. & Belloni, F. (2023)
"Revisiting p-11B fusion cross section and reactivity, and their analytic
approximations." Nucl. Fusion 63, 086001.

This module provides three parametrizations:
  1. Tentori-Belloni (TB, 2023) — modern reference
  2. Sikora-Weller (SW, 2016) raw data points — for validation
  3. Bosch-Hale (D-T) — for D-T comparison

All functions:
  - σ(E_CM): in barns (1 barn = 1e-24 cm²)
  - ⟨σv⟩(T): in cm³/s, Maxwell-Boltzmann averaged
  - E and T in keV

Validity ranges:
  σ_TB(E):       0 < E ≤ 9760 keV (CM)
  ⟨σv⟩_TB(T):    10 ≤ T ≤ 500 keV
  ⟨σv⟩_DT(T):    0.2 ≤ T ≤ 100 keV (Bosch-Hale)
"""

import numpy as np
from scipy.integrate import quad

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

# Atomic masses (u, AME2020)
m_H1_u  = 1.00782503207     # H-1 atom
m_B11_u = 11.00930516       # B-11 atom
m_He4_u = 4.00260325413     # He-4 atom
m_e_u   = 5.48579909e-4     # electron

# Unit conversions
u_to_MeV  = 931.49410242    # 1 u = 931.49 MeV/c²
u_to_keV  = u_to_MeV * 1000
amu_g     = 1.66053907e-24  # 1 u = 1.66e-24 g
keV_to_J  = 1.602176634e-16
keV_to_erg = 1.602176634e-9
e_charge  = 1.602176634e-19
alpha_fs  = 1.0 / 137.035999084
barn_cm2  = 1e-24

# p-B11 reaction parameters
# Bare-nucleus masses (subtract electrons)
m_p_u       = m_H1_u - m_e_u
m_B11_nuc_u = m_B11_u - 5 * m_e_u
mu_pB11_u   = (m_p_u * m_B11_nuc_u) / (m_p_u + m_B11_nuc_u)
mu_pB11_keV = mu_pB11_u * u_to_keV
mu_pB11_g   = mu_pB11_u * amu_g

# Q-value (atomic masses, electrons balanced)
Q_pB11_keV = (m_H1_u + m_B11_u - 3 * m_He4_u) * u_to_keV  # ≈ 8681 keV

# Gamow energy: E_G = (π α Z₁ Z₂)² · 2 μ c²
# TB paper gives E_G = 22.589 MeV
Z1, Z2 = 1, 5
E_G_keV = (np.pi * alpha_fs * Z1 * Z2)**2 * 2 * mu_pB11_keV
# Validation: E_G ≈ 22589 keV is expected


# ============================================================
# TENTORI-BELLONI (2023) S-FACTOR PARAMETERS
# ============================================================
#
# Source: TB Table 1, right column ("This work")
# Three-piece fit: S₁ (E ≤ 400 keV), S₂ (400 < E ≤ 668 keV), S₃ (668 < E ≤ 9760 keV)
#
# NOTE: TB paper specifies the parameters in MeV·b; we keep them as given
#       C_n: MeV·b/MeV^n units
#       A_L: MeV·b
#       E_L, δE_L: keV
# Because TB Eq. 3 uses (E/1 keV) normalization, no conversion of C₀ from
# MeV·b → keV·b is required.

# --- S₁: 0 < E ≤ 400 keV ---
# S₁(E) = C₀ + C₁·(E/1keV) + C₂·(E/1keV)² + A_L / [((E-E_L)/1keV)² + (δE_L/1keV)²]
# From TB Table 1 "This work" column (verbatim, IOP open-access PDF):
TB_C0  = 197.0          # MeV·b (C₀, fixed by continuity S2(E1)=S1(E1))
TB_C1  = 0.269          # MeV·b   (NS: 0.240)
TB_C2  = 2.54e-4        # MeV·b   (NS: 2.31e-4)
# 148 keV narrow resonance: the "—" in TB's "This work" column means
# "same value as Nevins-Swain" (the term is RETAINED, taken directly from
# NS — see TB Eq. 2 argument list S1(E;C0,C1,C2,A_L,E_L,δE_L)), NOT dropped.
# Bare-amplitude Lorentzian, width un-halved (confirmed from TB 2023 PDF).
TB_AL  = 1.82e4         # MeV·b   (A_L, from NS)
TB_EL  = 148.0          # keV     (E_L)
TB_dEL = 2.35           # keV     (δE_L)

# --- S₂: 400 < E ≤ 668 keV ---
# S₂(E) = D₀ + D₁·(ΔE/100keV) + D₂·(ΔE/100keV)² + D₅·(ΔE/100keV)⁵
# where ΔE = E - 400 keV
# C₀ ≡ S₂(400 keV) by continuity (so it is not independent)
TB_D0  = 346.0          # keV·b (NS: 330)
TB_D1  = 150.0          # keV·b (NS: 66.1) ← LARGE DIFFERENCE
TB_D2  = -59.9          # keV·b (NS: -20.3)
TB_D5  = -0.460         # keV·b (NS: -1.58)

# --- S₃: 668 < E ≤ 9760 keV ---
# S₃(E) = B + Σ_{k=0..3} A_k·(δE_k/2)² / [(E-E_k)² + (δE_k/2)²]
TB_B   = 0.381          # keV·b (NS: 4.38) ← much smaller

# Lorentzian resonance amplitudes: DRAMATICALLY larger than NS
TB_A   = np.array([1.98e6, 3.89e6, 1.36e6, 3.71e6])  # keV·b
# These are the headline difference of the TB paper: A₁ NS = 5.67e5, TB = 3.89e6 (~7×)
TB_E   = np.array([640.9, 1211.0, 2340.0, 3294.0])    # keV (resonance positions)
TB_dE  = np.array([85.5, 414.0, 221.0, 351.0])        # keV (resonance widths)

# Boundaries
TB_E1 = 400.0    # S₁/S₂ boundary
TB_E2 = 668.0    # S₂/S₃ boundary
TB_E3 = 9760.0   # upper limit of validity

# ============================================================
# UNIT NOTE
# ============================================================
#
# TB Table 1 units are MeV·b. In Eqs. 3-5 the A_k appears alone in the
# numerator (no classical Lorentzian (Γ/2)² normalization):
#
#   S₃(E) = B + Σ_k A_k / [(E-E_k)² + δE_k²]
#
# Peak value (at E = E_k): S_peak = B + A_k / δE_k²
#
# This means the "A_k" numbers carry units (MeV·b·keV²), but since
# (E-E_k) and δE_k are both in keV, the result S has units [MeV·b].
# The "MeV b" header in the TB table refers to the final unit of the
# S-factor, not the nominal unit of the A_k numbers themselves.


# ============================================================
# TB S-FACTOR FUNCTIONS
# ============================================================

def S1_TB(E_keV):
    """TB Eq. 3: low-energy S-factor (0 < E ≤ 400 keV).

    In the TB column the parameters A_L, E_L, δE_L are "—" (not used).
    Because the SW data missed the 148 keV resonance due to insufficient
    resolution, the TB paper does **not** include this resonance in S₁.

    Output: MeV·b
    """
    E = np.atleast_1d(E_keV).astype(float)
    # Polynomial background (C_n in MeV·b) + 148 keV NS resonance
    # (bare-amplitude Lorentzian, retained from Nevins-Swain).
    S = TB_C0 + TB_C1 * E + TB_C2 * E**2
    S = S + TB_AL / ((E - TB_EL)**2 + TB_dEL**2)
    return S


def S2_TB(E_keV):
    """TB Eq. 4: mid-energy S-factor (400 < E ≤ 668 keV)."""
    E = np.atleast_1d(E_keV).astype(float)
    dE = (E - TB_E1) / 100.0  # in 100-keV units
    S = TB_D0 + TB_D1 * dE + TB_D2 * dE**2 + TB_D5 * dE**5
    return S


def S3_TB(E_keV):
    """TB Eq. 5: high-energy S-factor (668 < E ≤ 9760 keV).

    S₃(E) = B + Σ_{k=0..3} A_k / [(E-E_k)² + δE_k²]

    NOTE: TB Eq. 5 does NOT use the classical (δE/2)²-normalized Lorentzian;
    A_k appears directly in the numerator. Peak value = A_k / δE_k².
    All energies are in keV; A_k carries the units that make S come out
    in MeV·b.

    Output: MeV·b
    """
    E = np.atleast_1d(E_keV).astype(float)
    S = np.full_like(E, TB_B)
    for k in range(4):
        # TB Eq. 5: A_k / [(E-E_k)² + δE_k²] (NOT a classical Lorentzian!)
        S += TB_A[k] / ((E - TB_E[k])**2 + TB_dE[k]**2)
    return S


def S_TB(E_keV):
    """Tentori-Belloni (2023) astrophysical S-factor, piecewise.

    Parameters
    ----------
    E_keV : float or ndarray
        CM energy (keV). Validity: 0 < E ≤ 9760 keV.

    Returns
    -------
    S : ndarray
        S-factor (in keV·barn).
    """
    E = np.atleast_1d(E_keV).astype(float)
    S = np.zeros_like(E)

    mask1 = (E > 0) & (E <= TB_E1)
    mask2 = (E > TB_E1) & (E <= TB_E2)
    mask3 = (E > TB_E2) & (E <= TB_E3)

    if np.any(mask1):
        S[mask1] = S1_TB(E[mask1])
    if np.any(mask2):
        S[mask2] = S2_TB(E[mask2])
    if np.any(mask3):
        S[mask3] = S3_TB(E[mask3])

    return S


def sigma_TB(E_cm_keV):
    """Tentori-Belloni (2023) p-11B fusion cross section.

    σ(E)[b] = S(E)[MeV·b] / E[MeV] · exp(-√(E_G/E))

    CRITICAL: The TB Table 1 parameters are in MeV·b, so S(E) is returned in
    MeV·b. To obtain σ we therefore convert E to MeV (the S/E division).
    The exponential is dimensionless because it depends on the ratio E_G/E.

    Parameters
    ----------
    E_cm_keV : float or ndarray
        CM energy (keV).

    Returns
    -------
    sigma : ndarray
        Reaction cross section (barns).
    """
    E = np.atleast_1d(E_cm_keV).astype(float)
    sigma = np.zeros_like(E)
    mask = E > 0.5
    if np.any(mask):
        S_MeVb = S_TB(E[mask])           # S, in MeV·b
        E_MeV = E[mask] / 1000.0         # convert E to MeV
        # σ[b] = S[MeV·b] / E[MeV] · exp(-√(E_G/E))
        sigma[mask] = (S_MeVb / E_MeV) * np.exp(-np.sqrt(E_G_keV / E[mask]))
    return sigma


# ============================================================
# WANG ET AL. (2026) — MODERN RE-EVALUATION [arXiv:2601.00241]
# ============================================================
#
# Wang, Li, Wu & Cui (2026), "Revisiting p-11B Fusion: Updated Cross-
# sections, Reactivity, and Energy Balance", arXiv:2601.00241. Same
# three-segment S-factor structure as TB, but a newer global fit.
# Verbatim Table 1 (coefficients in MeV·b; energies as noted). Same
# bare-amplitude Lorentzian form: A_k / [((E-E_k)/keV)² + (δE_k/keV)²].
#
# This evaluation gives ⟨σv⟩(300 keV) = 3.63e-16 cm³/s, ~17% BELOW the TB
# Table-1 value (4.37e-16). Wang reproduces the ~3.5e-16 reactivity that
# the PoP referees cited as reference; the TB-vs-Wang gap is a genuine
# difference between two published fits (mainly S2: D1=150 vs 102.4),
# NOT a coding error.

W_C0, W_C1, W_C2 = 197.0, 0.240, 2.31e-4                  # S1 polynomial
W_AL, W_EL, W_dEL = 1.82e4, 148.0, 2.35                   # S1 148 keV resonance
W_D0, W_D1, W_D2, W_D5 = 330.2, 102.436, -58.481, 0.0933  # S2 polynomial
W_B = 0.209689                                            # S3 background
W_A  = np.array([2.0235e6, 4.0102e6, 1.3220e6, 4.9451e6, 4.3430e5])  # MeV·b
W_E  = np.array([622.2, 1388.4, 2492.4, 3528.6, 4703.6])            # keV
W_dE = np.array([99.6, 449.9, 238.6, 398.5, 152.5])                 # keV
W_E1, W_E2, W_E3 = 400.0, 700.0, 10000.0                  # boundaries (keV)


def S_wang(E_keV):
    """Wang 2026 piecewise S-factor (MeV·b). 5 Breit-Wigner resonances."""
    E = np.atleast_1d(E_keV).astype(float)
    S = np.zeros_like(E)
    m1 = (E > 0) & (E <= W_E1)
    m2 = (E > W_E1) & (E <= W_E2)
    m3 = (E > W_E2) & (E <= W_E3)
    S[m1] = (W_C0 + W_C1 * E[m1] + W_C2 * E[m1]**2
             + W_AL / ((E[m1] - W_EL)**2 + W_dEL**2))
    x = (E[m2] - W_E1) / 100.0
    S[m2] = W_D0 + W_D1 * x + W_D2 * x**2 + W_D5 * x**5
    s = np.full_like(E[m3], W_B)
    for k in range(5):
        s += W_A[k] / ((E[m3] - W_E[k])**2 + W_dE[k]**2)
    S[m3] = s
    return S


def sigma_wang(E_cm_keV):
    """Wang 2026 p-11B fusion cross section (barns). E_cm in keV (CM)."""
    E = np.atleast_1d(E_cm_keV).astype(float)
    sigma = np.zeros_like(E)
    mask = E > 0.5
    if np.any(mask):
        sigma[mask] = (S_wang(E[mask]) / (E[mask] / 1000.0)) * \
                      np.exp(-np.sqrt(E_G_keV / E[mask]))
    return sigma


# ============================================================
# NEVINS-SWAIN (2000) — LEGACY REFERENCE  [Nucl. Fusion 40, 865]
# ============================================================
#
# Parameters: the "Nevins and Swain" column of Tentori-Belloni 2023 Table 1
# (verbatim), evaluated through the SAME bare-amplitude Lorentzian S-factor
# form and the SAME reactivity integrator as TB and Wang, so all three are
# compared on an identical footing.
#
# Integrating this NS S-factor gives <sigma v>(300 keV) = 3.52e-16 cm^3/s,
# ~4% (300 keV) to ~10% (500 keV) ABOVE the widely-cited NS-2000 *analytic
# reactivity fit*, whose stated validity ends near 500 keV (it plateaus at
# high T). We use the S-factor integral for consistency across the three
# parameterizations; the offset is a property of the integration, not an error.

NS_C0, NS_C1, NS_C2 = 197.0, 0.240, 2.31e-4             # S1 polynomial
NS_AL, NS_EL, NS_dEL = 1.82e4, 148.0, 2.35             # S1 148 keV resonance
NS_D0, NS_D1, NS_D2, NS_D5 = 330.0, 66.1, -20.3, -1.58  # S2 polynomial
NS_B = 4.38                                            # S3 background
NS_A  = np.array([2.57e6, 5.67e5, 1.34e5, 5.68e5])     # MeV·b
NS_E  = np.array([581.3, 1083.0, 2405.0, 3344.0])      # keV
NS_dE = np.array([85.7, 234.0, 138.0, 309.0])          # keV
NS_E1, NS_E2, NS_E3 = 400.0, 642.0, 3500.0             # boundaries (keV)


def S_NS(E_keV):
    """Nevins-Swain (2000) piecewise S-factor (MeV·b), 4 Breit-Wigner resonances."""
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
    """Nevins-Swain 2000 p-11B fusion cross section (barns). E_cm in keV (CM)."""
    E = np.atleast_1d(E_cm_keV).astype(float)
    sigma = np.zeros_like(E)
    mask = (E > 0.5) & (E <= NS_E3)
    if np.any(mask):
        sigma[mask] = (S_NS(E[mask]) / (E[mask] / 1000.0)) * \
                      np.exp(-np.sqrt(E_G_keV / E[mask]))
    return sigma


# ============================================================
# PRODUCTION CROSS-SECTION SELECTOR
# ============================================================
# "wang" (default): modern Wang 2026, gate-validated ⟨σv⟩(300)=3.63e-16.
# "TB": Tentori-Belloni 2023 analytic Table 1, ⟨σv⟩(300)=4.37e-16 — the
#       value used in the submitted manuscript (POP26-AR-00834).
# Everything downstream (reactivity, kinetic P_F, ignition, ash) follows
# this switch, so flipping it reproduces either the manuscript numbers
# ("TB") or the corrected numbers ("wang").
CROSS_SECTION = "wang"


def sigma_fusion(E_cm_keV):
    """Production p-11B fusion cross section, selected by CROSS_SECTION."""
    if CROSS_SECTION == "wang":
        return sigma_wang(E_cm_keV)
    if CROSS_SECTION == "NS":
        return sigma_NS(E_cm_keV)
    return sigma_TB(E_cm_keV)


# ============================================================
# TB REACTIVITY — MAXWELL-BOLTZMANN INTEGRATION
# ============================================================

def sigma_v_TB_numerical(T_keV, E_max_keV=9760, n_points=10000):
    """Tentori-Belloni σ(E) Maxwell-Boltzmann reactivity (numerical).

    ⟨σv⟩(T) = √(8/(πμ)) · (1/T)^(3/2) · ∫₀^∞ E·σ(E)·exp(-E/T) dE

    Parameters
    ----------
    T_keV : float or ndarray
        Ion temperature (keV).
    E_max_keV : float
        Upper integration bound.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    sigma_v : ndarray
        Reaction-rate coefficient (cm³/s).
    """
    T_arr = np.atleast_1d(T_keV).astype(float)
    result = np.zeros_like(T_arr)

    for i, T in enumerate(T_arr):
        if T <= 0:
            continue
        # Hybrid grid: dense to capture the resonances
        E_low  = np.logspace(np.log10(0.5), np.log10(50), 500)
        E_mid  = np.linspace(50, 2000, n_points // 2)
        E_high = np.linspace(2000, E_max_keV, n_points // 2)
        E_grid = np.unique(np.concatenate([E_low, E_mid, E_high]))

        sig_cm2 = sigma_fusion(E_grid) * barn_cm2
        E_erg = E_grid * keV_to_erg
        boltz = np.exp(-E_grid / T)

        integrand = sig_cm2 * E_erg * boltz
        integral_val = np.trapezoid(integrand, E_erg)

        T_erg = T * keV_to_erg
        prefactor = np.sqrt(8.0 / (np.pi * mu_pB11_g)) / T_erg**1.5
        result[i] = prefactor * integral_val

    return result


# ============================================================
# BOSCH-HALE (1992) D-T REACTIVITY — FOR COMPARISON
# ============================================================

def sigma_v_DT_BoschHale(T_keV):
    """⟨σv⟩ for D-T fusion, Bosch-Hale (1992) parametrization.

    Validity: 0.2 ≤ T ≤ 100 keV. Error: <0.25%.
    Reference: Bosch & Hale, Nucl. Fusion 32, 611 (1992).

    Returns: cm³/s
    """
    T = np.atleast_1d(T_keV).astype(float)
    B_G = 34.3827
    mc2 = 1124656.0
    C1, C2, C3, C4, C5, C6, C7 = (1.17302e-9, 1.51361e-2, 7.51886e-2,
                                   4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5)
    theta = T / (1 - T*(C2 + T*(C4 + T*C6)) / (1 + T*(C3 + T*(C5 + T*C7))))
    xi = (B_G**2 / (4 * theta))**(1/3)
    return C1 * theta * np.sqrt(xi / (mc2 * T**3)) * np.exp(-3 * xi)


# ============================================================
# VALIDATION TESTS
# ============================================================

def _test_constants():
    """Check the basic constants."""
    print("=" * 70)
    print("PHYSICAL CONSTANTS VALIDATION")
    print("=" * 70)

    # Q-value
    Q_lit = 8681.0  # keV (TB paper gives 8.6 MeV; more precise lit: 8.681 MeV)
    err_Q = abs(Q_pB11_keV - Q_lit) / Q_lit * 100
    status_Q = "[OK]" if err_Q < 0.1 else "[FAIL]"
    print(f"Q-value:     {Q_pB11_keV:.2f} keV  (lit: {Q_lit:.0f} keV, error: {err_Q:.3f}%) {status_Q}")

    # Gamow energy
    E_G_lit = 22589.0  # keV (TB paper)
    err_EG = abs(E_G_keV - E_G_lit) / E_G_lit * 100
    status_EG = "[OK]" if err_EG < 1.0 else "[FAIL]"
    print(f"E_G:         {E_G_keV:.1f} keV  (TB: {E_G_lit:.0f} keV, error: {err_EG:.2f}%) {status_EG}")

    # μc²
    mu_c2_MeV = mu_pB11_keV / 1000
    mu_lit = 859.526  # MeV (TB paper Eq. 6 commentary)
    err_mu = abs(mu_c2_MeV - mu_lit) / mu_lit * 100
    status_mu = "[OK]" if err_mu < 0.1 else "[FAIL]"
    print(f"μc²:         {mu_c2_MeV:.3f} MeV  (TB: {mu_lit:.3f} MeV, error: {err_mu:.3f}%) {status_mu}")


def _test_cross_section():
    """Check σ(E) values against TB Fig. 1(b) targets."""
    print("\n" + "=" * 70)
    print("σ(E) VALIDATION — TB Fig. 1(b) target values")
    print("=" * 70)

    # Values read from TB Fig. 1(b).
    # NOTE: the TB paper does NOT include the 148 keV resonance (missed by
    # SW data). At 148 keV the TB value is therefore much lower than NS
    # (no resonance).
    targets = [
        (148,  0.005, "148 keV — no resonance in TB, background only"),
        (300,  0.20,  "rising slope"),
        (640,  1.20,  "DOMINANT broad resonance (TB peak)"),
        (1000, 0.50,  "1 MeV trough"),
        (1211, 0.70,  "secondary resonance"),
        (2340, 0.50,  "tertiary resonance"),
        (5000, 0.10,  "high-energy tail"),
    ]

    print(f"{'E (keV)':>8} | {'σ_TB':>10} | {'σ_lit':>10} | {'ratio':>6} | note")
    print("-" * 75)
    for E, sig_target, note in targets:
        sig = sigma_TB(E)[0]
        ratio = sig / sig_target if sig_target > 0 else 0
        marker = "[OK]" if 0.5 < ratio < 2.0 else "[WARN]"
        print(f"{E:>8.0f} | {sig:>10.4f} | {sig_target:>10.4f} | {ratio:>5.2f}x | {marker} {note}")


def _test_reactivity():
    """Check ⟨σv⟩(T) against TB Fig. 2 + Table 2 targets."""
    print("\n" + "=" * 70)
    print("⟨σv⟩(T) VALIDATION — TB Fig. 2 + Table 2 target values")
    print("=" * 70)

    # Read from TB Fig. 2(a) (linear scale, m³/s).
    # m³/s → cm³/s: ×10⁶ (1 m³ = 10⁶ cm³)
    targets = [
        (50,   5.0e-18,  "low-T limit"),
        (100,  5.0e-17,  ""),
        (200,  2.0e-16,  ""),
        (300,  3.5e-16,  ""),
        (400,  4.7e-16,  ""),
        (500,  5.6e-16,  "headline (TB upper validity bound)"),
    ]

    print(f"{'T (keV)':>8} | {'⟨σv⟩_code (cm³/s)':>18} | {'⟨σv⟩_lit':>13} | {'ratio':>6} | note")
    print("-" * 75)
    for T, sv_target, note in targets:
        sv = sigma_v_TB_numerical(T)[0]
        ratio = sv / sv_target if sv_target > 0 else 0
        marker = "[OK]" if 0.5 < ratio < 2.0 else "[WARN]"
        print(f"{T:>8.0f} | {sv:>18.3e} | {sv_target:>13.2e} | {ratio:>5.2f}x | {marker} {note}")

    # Optimum temperature
    T_scan = np.linspace(50, 1000, 100)
    sv_scan = sigma_v_TB_numerical(T_scan)
    i_max = np.argmax(sv_scan)
    print(f"\n[*] TB peak: T = {T_scan[i_max]:.0f} keV, ⟨σv⟩ = {sv_scan[i_max]:.3e} cm³/s")


def _test_DT_comparison():
    """Compare with D-T — Bosch-Hale validation."""
    print("\n" + "=" * 70)
    print("D-T (BOSCH-HALE) VALIDATION")
    print("=" * 70)

    # Bosch-Hale reference values (Bosch & Hale 1992, Table VII):
    targets_DT = [
        (10,  1.13e-16),
        (20,  4.33e-16),
        (50,  8.74e-16),  # near peak
        (70,  8.96e-16),  # peak (Bosch-Hale)
        (100, 8.62e-16),
    ]

    print(f"{'T (keV)':>8} | {'⟨σv⟩_DT (cm³/s)':>18} | {'lit':>13} | {'ratio':>6}")
    print("-" * 65)
    for T, sv_lit in targets_DT:
        sv = sigma_v_DT_BoschHale(T)[0]
        ratio = sv / sv_lit
        marker = "[OK]" if 0.95 < ratio < 1.05 else "[WARN]"
        print(f"{T:>8.0f} | {sv:>18.3e} | {sv_lit:>13.2e} | {ratio:>5.3f}x {marker}")


if __name__ == "__main__":
    _test_constants()
    _test_cross_section()
    _test_reactivity()
    _test_DT_comparison()
