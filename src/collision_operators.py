"""
collision_operators.py — Trubnikov collision operators

Primary reference: Putvinski et al. (2019), Appendix A, Eqs. A.12-A.24
Trubnikov B.A. (1965), "Particle Interactions in a Fully Ionized Plasma"

This module provides the diffusion (D) and friction (F) coefficients for
an isotropic proton test particle. Field species: thermal protons (p),
boron (B), electrons (e), and (later) non-Maxwellian alpha particles (α*).

Putvinski Eq. 3:
  (1/v²)·∂_v[v²·(D_p·∂_v f_p + (F_p/m_p)·f_p)] - ν_fus·f_p + S = 0

where:
  D_p = D_pp + D_pB + D_pe + D*_pα   (Eq. 4)
  F_p = m_p²·v·[(D_pp + D_pB)/T + D_pe/T_e] + F*_pα   (Eq. 5)

CRITICAL VALIDATION: For a Maxwell distribution:
  D·∂_v f_M + (F/m)·f_M = 0   (detailed balance)
because for f_M = exp(-mv²/2T) we have ∂_v f_M = -(mv/T)·f_M,
so D·(-mv/T) + F/m = 0 → F = m²v·D/T  (in terms of T).

Each test particle in this module must satisfy detailed balance.

Unit system: CGS (cm, g, s, erg) — Trubnikov's original system.
"""

import numpy as np
from cross_sections import (
    mu_pB11_g, m_p_u, m_B11_nuc_u, amu_g,
    keV_to_erg, e_charge,
)

# ============================================================
# CGS UNITS AND PHYSICAL CONSTANTS
# ============================================================

# Basic CGS constants
m_p_g = m_p_u * amu_g           # proton mass, gram
m_B_g = m_B11_nuc_u * amu_g     # B-11 nuclear mass, gram
m_e_g = 9.1093837015e-28        # electron mass, gram
e_esu = 4.80320425e-10          # electron charge, esu (CGS Gauss)
c_cm  = 2.99792458e10           # speed of light, cm/s

# Charge states
Z_p = 1   # proton
Z_B = 5   # B-11
Z_e = -1  # electron (absolute value used because it appears squared)


# ============================================================
# COULOMB LOGARITHM
# ============================================================

def coulomb_log(n_e_cm3, T_e_keV, species_pair='ee'):
    """
    Coulomb logarithm Λ. NRL Plasma Formulary 2019, page 34.

    Parameters
    ----------
    n_e_cm3 : float
        Electron density, cm⁻³
    T_e_keV : float
        Electron temperature, keV
    species_pair : str
        'ee' (e-e), 'ei' (e-i), or 'ii' (i-i)

    Returns
    -------
    Lambda : float
        Coulomb logarithm (dimensionless)

    Notes
    -----
    Typical value for p-B11 plasmas is Λ ≈ 15-20.
    Putvinski takes Λ as a single parameter throughout the paper (specifies
    rather than computes it). We use the NRL formulas here, but it can be
    overridden with a fixed value (e.g. 17) where appropriate.
    """
    T_e_eV = T_e_keV * 1000.0

    if species_pair == 'ee':
        # e-e: valid for T_e ≥ 10 eV
        Lambda = 23.5 - np.log(n_e_cm3**0.5 * T_e_eV**(-1.25)) \
                 - np.sqrt(1e-5 + (np.log(T_e_eV) - 2)**2 / 16.0)
    elif species_pair == 'ei':
        # e-i, T_e > 10 eV (non-relativistic)
        # Putvinski Eq. A.6: λ_ie typically ~15-20
        if T_e_eV < 10.0:
            Lambda = 23.0 - np.log(n_e_cm3**0.5 * Z_p * T_e_eV**(-1.5))
        else:
            Lambda = 24.0 - np.log(n_e_cm3**0.5 / T_e_eV)
    elif species_pair == 'ii':
        # i-i, simplified form (single species, Z₁Z₂=1)
        T_i_eV = T_e_eV  # assumption
        Lambda = 23.0 - np.log(2 * n_e_cm3**0.5 / T_i_eV**1.5)
    else:
        raise ValueError(f"Unknown species_pair: {species_pair}")

    # Guard against unphysically small values (weak-coupling regime limit)
    return max(Lambda, 1.0)


# ============================================================
# I_ROSENBLUTH INTEGRAL — Putvinski Eq. A.11
# ============================================================
# General form, but a closed-form expression exists for Maxwellian field
# particles.

def D_test_on_Maxwellian(v_cm_s, n_field_cm3, T_field_keV, m_field_g,
                          Z_test, Z_field, lnLambda):
    """
    Diffusion coefficient of a test particle on Maxwellian field particles.
    Putvinski Eq. A.15:

      D_ρσ = (4π·Λ·Z_ρ²·Z_σ²·e⁴) / (A_ρ²·A_σ·m_p³) · (4π·T_σ/v³) · ∫₀^v v'²·f_σ(v') dv'

    With f_σ Maxwellian, the integral has a closed form.

    Simpler form (Putvinski Eq. A.22 and surroundings):

      D_ρσ = (4π·Λ·Z_ρ²·Z_σ²·e⁴·n_σ·T_σ) / (m_ρ²·m_σ·v) · G(x)

    where G(x) is a special function with x = v/v_th_σ.

    This module uses the GENERAL form (Putvinski Eq. A.15 + A.21):

      D_ρσ ≈ (4π·Λ·Z_ρ²·Z_σ²·e⁴·n_σ·T_σ) / (m_ρ²·m_p²·v³) ·
             [v³ / (v³ + (3√π/4)·v_th,σ³)]

    This interpolation is valid for the test particle at all speeds (slow and
    fast limits both correctly recovered).

    Parameters
    ----------
    v_cm_s : ndarray
        Test particle speeds, cm/s
    n_field_cm3 : float
        Field particle density, cm⁻³
    T_field_keV : float
        Field particle temperature, keV
    m_field_g : float
        Field particle mass, gram
    Z_test, Z_field : int
        Charge states
    lnLambda : float
        Coulomb logarithm

    Returns
    -------
    D : ndarray
        Diffusion coefficient, cm²/s³ (in velocity space)
    """
    v = np.atleast_1d(v_cm_s).astype(float)

    # Thermal speed (Maxwell): v_th = √(2T/m)
    T_field_erg = T_field_keV * keV_to_erg
    v_th = np.sqrt(2 * T_field_erg / m_field_g)

    # Test particle mass (assumed proton — this module is for the isotropic
    # proton test particle)
    m_test_g = m_p_g

    # Prefactor
    prefactor = (4 * np.pi * lnLambda * (Z_test * Z_field * e_esu**2)**2 *
                 n_field_cm3 * T_field_erg) / (m_test_g**2 * m_field_g)

    # Velocity dependence: smooth interpolation (Putvinski Eq. A.21)
    # v³ / (v³ + (3√π/4)·v_th³)
    smooth_factor = v**3 / (v**3 + (3 * np.sqrt(np.pi) / 4) * v_th**3)

    # D = prefactor / v³ · smooth_factor = prefactor / (v³ + (3√π/4)·v_th³)
    D = prefactor / (v**3 + (3 * np.sqrt(np.pi) / 4) * v_th**3)

    return D


def F_test_on_Maxwellian(v_cm_s, n_field_cm3, T_field_keV, m_field_g,
                          Z_test, Z_field, lnLambda):
    """
    Friction coefficient of a test particle on Maxwellian field particles.

    Detailed balance (Putvinski Eq. 5):
      F_ρσ = m_ρ² · v · D_ρσ / T_σ

    This is required for exact Maxwell equilibrium.
    Putvinski Eqs. A.13 and A.14 are equivalent.

    Returns
    -------
    F : ndarray
        Friction coefficient, gram·cm/s² (force)
    """
    v = np.atleast_1d(v_cm_s).astype(float)
    m_test_g = m_p_g

    D = D_test_on_Maxwellian(v_cm_s, n_field_cm3, T_field_keV, m_field_g,
                              Z_test, Z_field, lnLambda)

    T_field_erg = T_field_keV * keV_to_erg
    F = m_test_g**2 * v * D / T_field_erg

    return F


# ============================================================
# COLLISIONS WITH ELECTRONS — High-velocity limit
# ============================================================
#
# Putvinski Eq. A.24: protons are MUCH SLOWER than the electron thermal
# speed (v << v_th_e), so the Maxwell-collision formula simplifies to
#
#   D_pe ≈ (8√π·Λ·e⁴·n_e) / (3·m_p²) · √(m_e/(2·T_e))
#
# This form is essentially v-independent (weak v dependence). Our general
# formula automatically captures this limit thanks to the large v_th_e
# (electrons are very fast).

def D_pp(v, n_p, T_p_keV, lnLambda=17.0):
    """Proton-proton diffusion (thermal protons assumed Maxwellian)."""
    return D_test_on_Maxwellian(v, n_p, T_p_keV, m_p_g, Z_p, Z_p, lnLambda)


def D_pB(v, n_B, T_B_keV, lnLambda=17.0):
    """Proton-boron diffusion."""
    return D_test_on_Maxwellian(v, n_B, T_B_keV, m_B_g, Z_p, Z_B, lnLambda)


def D_pe(v, n_e, T_e_keV, lnLambda=17.0):
    """Proton-electron diffusion (high-v_th_e limit captured automatically)."""
    return D_test_on_Maxwellian(v, n_e, T_e_keV, m_e_g, Z_p, 1, lnLambda)


def F_pp(v, n_p, T_p_keV, lnLambda=17.0):
    """Proton-proton friction."""
    return F_test_on_Maxwellian(v, n_p, T_p_keV, m_p_g, Z_p, Z_p, lnLambda)


def F_pB(v, n_B, T_B_keV, lnLambda=17.0):
    """Proton-boron friction."""
    return F_test_on_Maxwellian(v, n_B, T_B_keV, m_B_g, Z_p, Z_B, lnLambda)


def F_pe(v, n_e, T_e_keV, lnLambda=17.0):
    """Proton-electron friction.

    NOTE: this uses T_e (the third term in Putvinski Eq. 5).
    If T_e < T_p, then F_pe < F_pp (proportionally), so the rate at which
    energy is transferred to electrons slows down — this is the heart of
    the hot-ion mode.
    """
    return F_test_on_Maxwellian(v, n_e, T_e_keV, m_e_g, Z_p, 1, lnLambda)


# ============================================================
# TOTAL DIFFUSION AND FRICTION (alpha contribution excluded)
# ============================================================

def D_total_thermal(v, n_p, n_B, n_e, T_p_keV, T_B_keV, T_e_keV, lnLambda=17.0):
    """Total diffusion from thermal field particles (alpha excluded)."""
    return (D_pp(v, n_p, T_p_keV, lnLambda) +
            D_pB(v, n_B, T_B_keV, lnLambda) +
            D_pe(v, n_e, T_e_keV, lnLambda))


def F_total_thermal(v, n_p, n_B, n_e, T_p_keV, T_B_keV, T_e_keV, lnLambda=17.0):
    """Total friction from thermal field particles (alpha excluded).

    Putvinski Eq. 5:
      F_p = m_p²·v·[(D_pp + D_pB)/T_i + D_pe/T_e]

    This ensures that when T_i ≠ T_e (hot-ion mode) the electron friction
    enters with its own temperature.
    """
    return (F_pp(v, n_p, T_p_keV, lnLambda) +
            F_pB(v, n_B, T_B_keV, lnLambda) +
            F_pe(v, n_e, T_e_keV, lnLambda))


# ============================================================
# VALIDATION: DETAILED BALANCE TEST
# ============================================================

def _test_detailed_balance():
    """
    For a Maxwell distribution, D·∂_v f_M + (F/m)·f_M = 0 must hold.

    This verifies that the collision operator drives the system to Maxwell
    equilibrium. If the test fails, the FP solver will not preserve the
    Maxwell distribution and the code is fundamentally broken.
    """
    print("=" * 70)
    print("DETAILED BALANCE TEST")
    print("=" * 70)
    print("Under Maxwell f_M, expect: D·∂_v f_M + (F/m)·f_M = 0")
    print()

    # Parameters: typical p-B11 plasma conditions
    n_p = 8.5e13   # cm⁻³
    n_B = 1.5e13   # cm⁻³
    n_e = n_p + Z_B * n_B  # quasineutrality
    T_p = 300.0    # keV
    T_B = 300.0    # keV
    T_e = 150.0    # keV (hot-ion mode)
    lnLambda = 17.0

    # Velocity grid — around the thermal speed
    T_p_erg = T_p * keV_to_erg
    v_th_p = np.sqrt(2 * T_p_erg / m_p_g)  # cm/s
    print(f"v_th_p = {v_th_p:.3e} cm/s")

    v_grid = np.linspace(0.1, 5.0, 50) * v_th_p  # 0.1·v_th to 5·v_th

    # Maxwell distribution (at temperature T_p)
    f_M = (m_p_g / (2 * np.pi * T_p_erg))**1.5 * np.exp(-m_p_g * v_grid**2 / (2 * T_p_erg))

    # Derivative: ∂_v f_M = -(m·v/T)·f_M
    df_M_dv = -(m_p_g * v_grid / T_p_erg) * f_M

    # IMPORTANT: detailed balance is exact when T_p = T_B = T_e.
    # In hot-ion mode (T_e < T_p) the electron contribution unbalances it.

    # FIRST: SINGLE-TEMPERATURE TEST (T_p = T_B = T_e = 300 keV)
    print("\n--- Test 1: Single temperature (T_p = T_B = T_e = 300 keV) ---")
    D_total = D_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_p, lnLambda)
    F_total = F_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_p, lnLambda)

    # FP flux: J = D·∂_v f + (F/m)·f
    # Detailed balance: J = 0
    flux = D_total * df_M_dv + (F_total / m_p_g) * f_M

    # Relative error: |flux| / (D · |∂_v f_M|)
    rel_err = np.abs(flux) / (D_total * np.abs(df_M_dv) + 1e-300)

    print(f"Max relative error: {np.max(rel_err):.3e}")
    print(f"Mean relative error: {np.mean(rel_err):.3e}")

    if np.max(rel_err) < 1e-10:
        print("[OK] Detailed balance satisfied to MACHINE PRECISION at single T")
    elif np.max(rel_err) < 1e-3:
        print("[OK] Detailed balance satisfied at single T (within numerical error)")
    else:
        print("[FAIL] DETAILED BALANCE BROKEN — code is incorrect")

    # HOT-ION MODE TEST
    print("\n--- Test 2: Hot-ion mode (T_p = T_B = 300, T_e = 150 keV) ---")
    print("Detailed balance is NOT exactly satisfied here (T_e ≠ T_p).")
    print("FP flux: J = (mv·f) · [D_pe·(1/T_e - 1/T_p)]")
    print("If T_e < T_p then (1/T_e - 1/T_p) > 0 → flux POSITIVE (towards high v)")
    print("This causes the proton tail to GROW — the basis of Putvinski's")
    print("~10% kinetic fusion enhancement.")

    D_total = D_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_e, lnLambda)
    F_total = F_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_e, lnLambda)
    flux = D_total * df_M_dv + (F_total / m_p_g) * f_M

    # At high v the flux must be positive (proton tail accumulating)
    i_high_v = np.argmin(np.abs(v_grid - 3 * v_th_p))
    print(f"\nFlux at v = 3·v_th: {flux[i_high_v]:.3e}")
    if flux[i_high_v] > 0:
        print("[OK] POSITIVE flux — hot-ion mode pushes the proton tail upward")
        print("     (kinetic enhancement mechanism, Putvinski 2019 §2.1)")
    else:
        print("[FAIL] NEGATIVE flux — unexpected, physics is wrong")


def _test_individual_components():
    """Verify that each component (D_pp, D_pB, D_pe) scales correctly on its own."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL COMPONENT TEST")
    print("=" * 70)

    n_test = 1e14    # cm⁻³
    T_test = 300.0   # keV
    T_p_erg = T_test * keV_to_erg
    v_th_p = np.sqrt(2 * T_p_erg / m_p_g)
    v_test = v_th_p  # evaluate near thermal speed

    print(f"\nConditions: n = 10¹⁴ cm⁻³, T = 300 keV, v = v_th,p = {v_th_p:.2e} cm/s\n")

    # Putvinski's estimate (300 keV plasma: ν_ii ≈ 100 s⁻¹)
    # ν_pp ~ D_pp / v² (dimensional analysis)

    D_pp_val = D_pp(v_test, n_test, T_test)[0]
    D_pB_val = D_pB(v_test, n_test * 0.176, T_test)[0]  # n_B = 0.15·n_e
    D_pe_val = D_pe(v_test, n_test * 1.75, T_test)[0]   # n_e ≈ 1.75·n_p

    print(f"D_pp = {D_pp_val:.3e} cm²/s³")
    print(f"D_pB = {D_pB_val:.3e} cm²/s³")
    print(f"D_pe = {D_pe_val:.3e} cm²/s³")

    # Expectation: D_pB > D_pp (Z_B² = 25), D_pe << D_pp (m_e << m_p factor)
    ratio_BB_pp = D_pB_val / D_pp_val
    ratio_pe_pp = D_pe_val / D_pp_val

    print(f"\nD_pB / D_pp = {ratio_BB_pp:.2f}")
    print("  Expected: (Z_B²/A_B)·(n_B/n_p)·smoothing_correction")
    print("           = (25/11)·0.176·(1/0.69) ≈ 0.6")
    print(f"  {'[OK]' if 0.3 < ratio_BB_pp < 1.5 else '[WARN]'}")

    print(f"\nD_pe / D_pp = {ratio_pe_pp:.3e}")
    print("  Expected: (n_e/n_p)·(T_e/T_p)·(m_p/m_e)·smoothing ~ 0.05")
    print("  (At low v, v_th,e³ dominates the denominator, suppressing D_pe.)")
    print(f"  {'[OK]' if 0.01 < ratio_pe_pp < 0.5 else '[WARN]'}")

    # Collision frequency estimate
    nu_pp = D_pp_val / v_test**2
    print(f"\nν_pp ≈ D_pp/v² = {nu_pp:.3e} s⁻¹")
    print("  Putvinski 300 keV plasma typical: ν_ii ~ 10²-10³ s⁻¹")


if __name__ == "__main__":
    _test_detailed_balance()
    _test_individual_components()
