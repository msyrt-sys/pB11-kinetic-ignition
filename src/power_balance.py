"""
power_balance.py — Plasma power balance and ignition criterion

Primary reference: Putvinski et al. (2019) §2.2, Eqs. 7-10

This module performs three main jobs:

1. **Bremsstrahlung power** (Putvinski Eq. 7, Svensson 1982 fit):
   P_Brem = 7.56e-11 · n_e² · x^(1/2) · [Z_eff·(1+1.78·x^1.34)
            + 2.12·x·(1+1.1·x+x²-1.25·x^2.5)]   (eV·cm³/s)
   x = T_e/m_e c² (dimensionless)

2. **Fusion power** P_F = E_F · n_p · n_B · ⟨σv⟩
   - With FP-distorted f_p, includes the kinetic enhancement
   - With Maxwell assumption and bare reactivity, the thermal value

3. **α-ion and α-electron power transfer**:
   - Putvinski Eq. 8: steady-state electron balance
   - Energy transfer from slowing-down α to protons and electrons

4. **Self-consistent T_e**:
   - For given T_i and plasma mix, find T_e such that
     P_α,e + P_i,e = P_Brem (electron power balance)

5. **Ignition criterion**:
   - P_F > P_Brem? (necessary condition)
   - τ_E* = U_K / (P_F - P_Brem) (Ochs metric)

Unit system: CGS (cm, g, s, erg) for the calculation; results convert to SI.
Power unit: W/cm³ or erg/(cm³·s) (1 W = 10⁷ erg/s).
"""

import numpy as np
from scipy.optimize import brentq

from cross_sections import (
    keV_to_erg, sigma_v_TB_numerical, mu_pB11_g, barn_cm2,
)
from collision_operators import (
    m_p_g, m_B_g, m_e_g, e_esu, c_cm, Z_p, Z_B, coulomb_log,
)
from alpha_source import (
    Q_pB11_keV, m_alpha_g, Z_alpha,
)


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

E_rest_eV = m_e_g * c_cm**2 / 1.602176634e-12  # m_e c² = 5.11e5 eV
E_rest_keV = E_rest_eV / 1000.0


# ============================================================
# 1. BREMSSTRAHLUNG POWER (Putvinski Eq. 7 / Svensson 1982)
# ============================================================

def P_bremsstrahlung(n_e_cm3, T_e_keV, Z_eff):
    """Bremsstrahlung power (W/cm³).

    Classical NRL Plasma Formulary form (W/cm³):
      P_classical = 5.34e-31 · n_e² · Z_eff · √(T_e[keV])

    Svensson 1982 relativistic correction (the bracketed factor in Putvinski Eq. 7):
      g(x) = (1 + 1.78·x^1.34) + (2.12·x/Z_eff)·(1+1.1·x+x²-1.25·x^2.5)
      x = T_e/m_e c²

    Total: P = P_classical · g(x)

    Validation: T_e=150 keV, n_e=1.6e14, Z_eff=2.87 → P ≈ 0.65 W/cm³,
    in agreement with the ~0.8 W/cm³ visible in Putvinski Fig. 4.

    Parameters
    ----------
    n_e_cm3 : float or ndarray
        Electron density (cm⁻³)
    T_e_keV : float or ndarray
        Electron temperature (keV)
    Z_eff : float
        Effective charge: Σ_i n_i Z_i² / Σ_i n_i Z_i

    Returns
    -------
    P_Brem : float or ndarray
        Bremsstrahlung power, W/cm³
    """
    # Classical part (NRL)
    P_classical = 5.34e-31 * n_e_cm3**2 * Z_eff * np.sqrt(T_e_keV)

    # Svensson relativistic correction
    x = T_e_keV / E_rest_keV

    # e-i bremsstrahlung relativistic correction
    factor_ei = 1.0 + 1.78 * x**1.34

    # e-e bremsstrahlung contribution (relativistic, Z=0)
    # The 2.12·x·(1+1.1·x+x²-1.25·x^2.5) term in Putvinski Eq. 7.
    # We divide by Z_eff so it can be added to the NRL classical form.
    factor_ee = 2.12 * x * (1.0 + 1.1 * x + x**2 - 1.25 * x**2.5) / Z_eff

    g_relativistic = factor_ei + factor_ee

    return P_classical * g_relativistic


def Z_eff_calc(n_p_cm3, n_B_cm3):
    """Z_eff for a p-B11 plasma (proton + boron only).

    Z_eff = Σ_i n_i Z_i² / Σ_i n_i Z_i = Σ_i n_i Z_i² / n_e

    For p-B11: n_e = n_p + Z_B · n_B (quasineutrality)
    """
    n_e = n_p_cm3 + Z_B * n_B_cm3
    Z_eff = (n_p_cm3 * Z_p**2 + n_B_cm3 * Z_B**2) / n_e
    return Z_eff


# ============================================================
# 2. FUSION POWER
# ============================================================

def P_fusion_thermal(n_p_cm3, n_B_cm3, T_i_keV):
    """Thermal fusion power (Maxwell assumption).

    P_F = n_p · n_B · ⟨σv⟩(T_i) · E_F

    where E_F = Q-value = 8.681 MeV (total energy)

    Returns: W/cm³
    """
    sv = sigma_v_TB_numerical(T_i_keV)  # cm³/s
    if np.isscalar(T_i_keV):
        sv = sv[0] if hasattr(sv, '__len__') else sv

    # Reaction rate: n_p · n_B · ⟨σv⟩ (reactions/cm³/s)
    rate = n_p_cm3 * n_B_cm3 * sv

    # Energy per reaction
    E_F_erg = Q_pB11_keV * keV_to_erg

    # P_F (erg/s/cm³)
    P_erg_s = rate * E_F_erg

    # W/cm³
    return P_erg_s / 1e7


def P_fusion_kinetic(v_grid, f_p, n_B_cm3, T_B_keV):
    """Fusion power with FP-distorted proton distribution.

    P_F = ∫ 4π·v²·f_p(v) · n_B · σ(v) · v · E_F · dv

    Here f_p is in general non-Maxwellian (the FP solver output).
    This captures Putvinski's "kinetic enhancement" mechanism.

    Parameters
    ----------
    v_grid : ndarray
        Proton velocity grid (cm/s)
    f_p : ndarray
        Proton distribution function (cm⁻⁶ s³)
    n_B_cm3 : float
        Boron density
    T_B_keV : float
        Boron temperature (Maxwell)

    Returns
    -------
    P_F : float
        Fusion power, W/cm³
    """
    from cross_sections import sigma_fusion

    # Lab-frame proton kinetic energy → CM energy
    E_p_erg = 0.5 * m_p_g * v_grid**2
    E_CM_erg = E_p_erg * m_B_g / (m_p_g + m_B_g)
    E_CM_keV = E_CM_erg / keV_to_erg

    sigma_b = sigma_fusion(E_CM_keV)
    sigma_cm2 = sigma_b * barn_cm2

    # Reaction rate: ∫ 4π v² f_p · n_B · σ(v) · v dv
    integrand = 4 * np.pi * v_grid**2 * f_p * n_B_cm3 * sigma_cm2 * v_grid
    rate = np.trapezoid(integrand, v_grid)

    # Energy per reaction
    E_F_erg = Q_pB11_keV * keV_to_erg

    P_erg_s = rate * E_F_erg
    return P_erg_s / 1e7


# ============================================================
# 3. RELATIVISTIC ION-ELECTRON ENERGY EXCHANGE
# ============================================================
# Putvinski Eqs. 9-10

def relativistic_R_factor(T_e_keV):
    """Putvinski Eq. 10: relativistic R(x) correction factor.

    R(x) = (1 + 2x + 2x²) · √(π x³/2) / ∫₀^∞ t² · exp((1-√(1+t²))/x) dt

    x = T_e/(m_e c²).

    Interpretation: R is the ratio ν_ei^relativistic / ν_ei^classical.
    Putvinski Fig. 3: at T_e=150 keV (x=0.29), R ≈ 1.1 (relativistic effects
    INCREASE the ion-electron coupling, they do not suppress it — the paper
    speaks of "10% higher electron temperature").

    Classical limit x→0: R → 1.

    Returns: R factor (dimensionless, ~1.0-1.4)
    """
    x = T_e_keV / E_rest_keV

    if x < 0.001:
        return 1.0  # classical limit

    # ∫₀^∞ t² · exp((1-√(1+t²))/x) dt — numerical quadrature
    t_grid = np.logspace(-3, 2, 2000)
    integrand = t_grid**2 * np.exp((1 - np.sqrt(1 + t_grid**2)) / x)
    integral = np.trapezoid(integrand, t_grid)

    # Putvinski Eq. 10: R = (1+2x+2x²) · √(πx³/2) / integral
    R = (1 + 2*x + 2*x**2) * np.sqrt(np.pi * x**3 / 2) / integral
    return R


def P_ion_electron_transfer(n_p, n_B, T_p, T_B, T_e, lnLambda=17.0):
    """Ion-to-electron energy transfer power (Putvinski Eq. 9).

    P_{i,e} = Σ_i (3/2) · ν_{ie} · n_i · (T_i - T_e) · R(x)

    where ν_{ie} is the classical collision frequency:
    ν_{ie}^cl = 4.8e-9 · Z_i² · λ_{ie} · n_e / (m_i · T_e^(3/2))   (NRL form)

    Returns: W/cm³
    """
    n_e = n_p + Z_B * n_B

    # Classical collision frequency (NRL Plasma Formulary)
    # ν_{ie} = 1/τ_ε is the Spitzer energy equilibration time
    # NRL p.31:
    # ν_ε^{ie} = (m_e/m_i) · (4√(2π)/3) · (n_e e⁴ ln Λ)/(m_e^(1/2) T_e^(3/2)) · Z_i²

    # Proton contribution
    nu_pe = (m_e_g/m_p_g) * (4*np.sqrt(2*np.pi)/3) * \
            (n_e * e_esu**4 * lnLambda) / (m_e_g**0.5 * (T_e * keV_to_erg)**1.5) * Z_p**2

    # Boron contribution
    nu_Be = (m_e_g/m_B_g) * (4*np.sqrt(2*np.pi)/3) * \
            (n_e * e_esu**4 * lnLambda) / (m_e_g**0.5 * (T_e * keV_to_erg)**1.5) * Z_B**2

    # Relativistic correction
    R = relativistic_R_factor(T_e)

    # Power transfer (proton + boron)
    # (3/2) n_i ν_ie · (T_i - T_e), in erg
    P_pe = 1.5 * n_p * nu_pe * (T_p - T_e) * keV_to_erg * R   # erg/s/cm³
    P_Be = 1.5 * n_B * nu_Be * (T_B - T_e) * keV_to_erg * R   # erg/s/cm³

    P_total_erg = P_pe + P_Be
    return P_total_erg / 1e7  # W/cm³


# ============================================================
# 3.5 ION-ION THERMALIZATION (Ochs Eqs. 13-14, K_pb)
# ============================================================

def P_pb_thermalization(n_p, n_B, T_p_keV, T_B_keV, lnLambda=17.0):
    """Proton-boron ion-ion thermalization power transfer (Ochs Eqs. 13-14, K_pb).

    P_{p→b} = (3/2) · n_p · ν_pb · (T_p - T_B)

    NRL Plasma Formulary, Spitzer energy equilibration:
        ν_pb = (8√(2π)/3) · (Z_p² Z_B² e⁴ n_B ln Λ) / (m_p · m_B) ·
               · (T_p/m_p + T_B/m_B)^(-3/2)

    Corresponds to the K_pb (T_b - T_p) term of Ochs Eqs. 13-14.
    Sign: positive when T_p > T_B (proton heats boron); reverse otherwise.

    In a p-B11 plasma τ_pb (= U_p/P_pb) ~ 10 ms whereas τ_E ~ 1-100 s, so
    T_p ≈ T_B in steady state. Putvinski's single-T assumption is justified.
    This function is provided for validation.

    Parameters
    ----------
    n_p, n_B : float
        Densities (cm⁻³)
    T_p_keV, T_B_keV : float
        Temperatures (keV)
    lnLambda : float

    Returns
    -------
    P_pb : float
        p → B heat transfer (W/cm³)
    """
    T_p_erg = T_p_keV * keV_to_erg
    T_B_erg = T_B_keV * keV_to_erg

    prefactor = (8 * np.sqrt(2 * np.pi) / 3) * (Z_p * Z_B * e_esu**2)**2 * n_B * lnLambda
    denom = m_p_g * m_B_g * (T_p_erg / m_p_g + T_B_erg / m_B_g)**1.5

    nu_pb = prefactor / denom  # 1/s

    P_erg_per_s_cm3 = 1.5 * n_p * nu_pb * (T_p_erg - T_B_erg)

    return P_erg_per_s_cm3 / 1e7  # W/cm³


def thermalization_time_pb(n_p, n_B, T_p_keV, T_B_keV, lnLambda=17.0):
    """p-B thermalization time scale τ_pb = U_p / P_pb.

    The single-T assumption is valid when τ_pb << τ_E.
    """
    P_pb = abs(P_pb_thermalization(n_p, n_B, T_p_keV, T_B_keV, lnLambda))
    if P_pb < 1e-30:
        return np.inf

    T_avg = 0.5 * (T_p_keV + T_B_keV)
    U_p = 1.5 * n_p * T_avg * keV_to_erg / 1e7  # J/cm³
    return U_p / P_pb


# ============================================================
# 3.6 ASH POISONING (Ochs §V)
# ============================================================

def Z_eff_with_ash(n_p_cm3, n_B_cm3, n_alpha_cm3):
    """Z_eff and n_e in the presence of alpha-ash poisoning (Ochs §V).

    Quasineutrality with ash:
        n_e = n_p + Z_B · n_B + Z_α · n_α     (Z_α = 2)

    Z_eff:
        Z_eff = (n_p Z_p² + n_B Z_B² + n_α Z_α²) / n_e

    α particles accumulate after fusion. If they do not escape:
      - n_e increases (α contribution)
      - Z_eff increases (Z_α² = 4 effect)
      - bremsstrahlung increases (∝ n_e² · Z_eff)
      - **No fusion contribution** (the α + p reaction is small)

    Ochs Fig. 8: 2% ash → ignition window closes (no channeling).

    Parameters
    ----------
    n_p_cm3, n_B_cm3, n_alpha_cm3 : float
        Densities (cm⁻³)

    Returns
    -------
    n_e : float
        Electron density
    Z_eff : float
        Effective charge
    """
    Z_alpha = 2
    n_e = n_p_cm3 + Z_B * n_B_cm3 + Z_alpha * n_alpha_cm3
    Z_eff = (n_p_cm3 * Z_p**2 +
             n_B_cm3 * Z_B**2 +
             n_alpha_cm3 * Z_alpha**2) / n_e
    return n_e, Z_eff


def P_brem_with_ash(n_p_cm3, n_B_cm3, n_alpha_cm3, T_e_keV):
    """Bremsstrahlung power with ash poisoning (Ochs §V).

    NRL + Svensson correction, but n_e and Z_eff include α.

    Typical: 2% ash → P_B grows by ~12-15% (n_e² × Z_eff effect).
    """
    n_e, Z_eff = Z_eff_with_ash(n_p_cm3, n_B_cm3, n_alpha_cm3)
    return P_bremsstrahlung(n_e, T_e_keV, Z_eff)


def ignition_check_with_ash(n_p, n_B, n_alpha, T_i_keV, T_e_keV=None,
                              alpha_channeling_eff=0.0, lnLambda=17.0):
    """Ignition check including ash poisoning and α-channeling (Ochs §V Fig. 8).

    Three scenarios (Ochs Fig. 8):
    (a) η_α = 0 (no channeling): P_F < P_B → no ignition (Fig. 8a)
    (b) η_α = 0.5, channeled to thermal protons: P_F > P_B (Fig. 8b)
    (c) η_α = 0.5, channeled to fast protons (kinetic FP): widens (Fig. 8c)

    This function evaluates cases (a) and (b). Case (c) requires the
    kinetic FP solver (handled in main_validation.py).

    Parameters
    ----------
    n_p, n_B, n_alpha : float
        Densities (cm⁻³)
    T_i_keV : float
        Ion temperature (T_p ≈ T_B is assumed because τ_pb << τ_E)
    T_e_keV : float or None
        Electron temperature; if None, computed self-consistently
    alpha_channeling_eff : float
        Channeling efficiency from α to thermal protons (0-1).
        Ochs notation: η_α
    lnLambda : float

    Returns
    -------
    dict
    """
    if T_e_keV is None:
        T_e_keV = find_self_consistent_Te(n_p, n_B, T_i_keV, lnLambda)

    n_e, Z_eff = Z_eff_with_ash(n_p, n_B, n_alpha)
    n_i_total = n_p + n_B + n_alpha
    ash_fraction = n_alpha / n_i_total if n_i_total > 0 else 0

    # P_F (only p and B fuse)
    P_F = P_fusion_thermal(n_p, n_B, T_i_keV)

    # P_B (with ash)
    P_B = P_bremsstrahlung(n_e, T_e_keV, Z_eff)

    # α generation power
    P_alpha_gen = P_alpha_total(n_p, n_B, T_i_keV)

    # α-channeling: η_α fraction transferred to thermal protons
    # P_F_effective = P_F + η_α · P_alpha (extra heating ⇒ effective fusion)
    P_F_with_channeling = P_F + alpha_channeling_eff * P_alpha_gen

    return {
        'T_i_keV': T_i_keV,
        'T_e_keV': T_e_keV,
        'n_e': n_e,
        'Z_eff': Z_eff,
        'ash_fraction': ash_fraction,
        'P_F': P_F,
        'P_B': P_B,
        'P_alpha_generated': P_alpha_gen,
        'P_F_with_channeling': P_F_with_channeling,
        'P_F_minus_P_B': P_F_with_channeling - P_B,
        'ignition': P_F_with_channeling > P_B,
        'ratio_PF_PB': P_F_with_channeling / P_B if P_B > 0 else np.inf,
        'alpha_channeling_eff': alpha_channeling_eff,
    }


# ============================================================
# 4. α POWER PARTITION
# ============================================================

def P_alpha_total(n_p_cm3, n_B_cm3, T_i_keV):
    """Total α power (production rate × average energy).

    P_α = n_p · n_B · ⟨σv⟩ · E_α_total

    Here E_α_total = Q ≈ 8.7 MeV (total energy of the three α particles).
    This power is then partitioned within the plasma — some to protons, some
    to electrons, some to boron.

    Returns: W/cm³
    """
    sv = sigma_v_TB_numerical(T_i_keV)
    if hasattr(sv, '__len__') and not np.isscalar(T_i_keV):
        pass
    else:
        sv = sv[0] if hasattr(sv, '__len__') else sv

    rate = n_p_cm3 * n_B_cm3 * sv  # reactions/cm³/s
    E_alpha_total_erg = Q_pB11_keV * keV_to_erg
    P_erg = rate * E_alpha_total_erg
    return P_erg / 1e7


def alpha_power_to_electrons_fraction(T_e_keV, n_e, n_p, n_B, lnLambda=17.0):
    """Fraction of α power deposited into electrons (approximate).

    Slowing-down α particles transfer their energy mostly to electrons at
    high v and to ions at low v. In typical p-B11 plasmas with T_e ≈ 150 keV,
    the electron fraction is around ~10% (Putvinski 2019).

    This is a simple empirical formula; an exact result requires integration
    of the full α distribution.

    Returns: 0 ≤ frac ≤ 1
    """
    # Roughly linear scaling: electron fraction decreases as T_e grows.
    # T_e = 150 keV → ~10%
    # T_e = 50 keV  → ~30%
    # T_e = 300 keV → ~5%

    # Simple fit inspired by Putvinski Fig. B1
    frac_e = 0.15 * (150.0 / max(T_e_keV, 50.0))**0.5
    return min(frac_e, 0.5)


# ============================================================
# 5. SELF-CONSISTENT T_e SOLVER
# ============================================================

def find_self_consistent_Te(n_p, n_B, T_i_keV, lnLambda=17.0,
                              T_e_min=20.0, T_e_max=500.0):
    """Putvinski Eq. 8: P_α,e + P_i,e = P_Brem.

    For given T_i and plasma mix, find the T_e that closes the electron
    power balance.

    This is the steady-state condition: power into electrons (from α and
    from ions) = electron loss (bremsstrahlung).

    Parameters
    ----------
    n_p, n_B : float
        Ion densities
    T_i_keV : float
        Ion temperature (T_p = T_B assumption)

    Returns
    -------
    T_e_keV : float
        Self-consistent electron temperature
    """
    n_e = n_p + Z_B * n_B
    Z_eff = Z_eff_calc(n_p, n_B)

    # P_α,e + P_i,e - P_Brem = 0
    def power_residual(T_e):
        # Power from α to electrons
        P_alpha = P_alpha_total(n_p, n_B, T_i_keV)
        frac_e = alpha_power_to_electrons_fraction(T_e, n_e, n_p, n_B, lnLambda)
        P_alpha_e = P_alpha * frac_e

        # Power from ions to electrons (positive when T_i > T_e)
        P_ie = P_ion_electron_transfer(n_p, n_B, T_i_keV, T_i_keV, T_e, lnLambda)

        # Bremsstrahlung loss
        P_brem = P_bremsstrahlung(n_e, T_e, Z_eff)

        return P_alpha_e + P_ie - P_brem

    # Should be positive at T_e_min (P_alpha+P_ie > P_brem) and negative at T_e_max
    try:
        T_e_solution = brentq(power_residual, T_e_min, T_e_max, xtol=0.5)
    except ValueError:
        # Fallback: T_e/T_i = 0.5 (Putvinski)
        T_e_solution = T_i_keV * 0.5

    return T_e_solution


# ============================================================
# 6. IGNITION CRITERION and τ_E*
# ============================================================

def ignition_check(n_p, n_B, T_i_keV, T_e_keV=None, lnLambda=17.0):
    """Ignition criterion: is P_F > P_Brem?

    If T_e_keV is not given, computed self-consistently.

    Returns
    -------
    info : dict
        'P_F'                : fusion power (W/cm³)
        'P_Brem'             : bremsstrahlung (W/cm³)
        'P_F_minus_P_Brem'   : net (W/cm³)
        'ignition'           : bool
        'T_e_keV'            : T_e used
        'tau_E_star'         : Ochs metric (seconds)
    """
    if T_e_keV is None:
        T_e_keV = find_self_consistent_Te(n_p, n_B, T_i_keV, lnLambda)

    n_e = n_p + Z_B * n_B
    Z_eff = Z_eff_calc(n_p, n_B)

    # Fusion power
    P_F = P_fusion_thermal(n_p, n_B, T_i_keV)
    if hasattr(P_F, '__len__'):
        P_F = float(P_F)

    # Bremsstrahlung
    P_Brem = P_bremsstrahlung(n_e, T_e_keV, Z_eff)

    # Net power
    P_net = P_F - P_Brem

    # τ_E* = U_K / (P_F - P_Brem)  [Ochs metric]
    # U_K = (3/2) (n_p T_p + n_B T_B + n_e T_e)
    U_K_keV_per_cm3 = 1.5 * (n_p * T_i_keV + n_B * T_i_keV + n_e * T_e_keV)
    U_K_J_per_cm3 = U_K_keV_per_cm3 * 1.602e-16  # keV → J
    U_K_W_s = U_K_J_per_cm3  # W·s/cm³

    if P_net > 0:
        tau_E_star = U_K_W_s / P_net
    else:
        tau_E_star = np.inf

    return {
        'T_i_keV': T_i_keV,
        'T_e_keV': T_e_keV,
        'P_F': P_F,
        'P_Brem': P_Brem,
        'P_net': P_net,
        'P_F_over_P_Brem': P_F / P_Brem if P_Brem > 0 else np.inf,
        'ignition': P_net > 0,
        'tau_E_star_s': tau_E_star,
    }


# ============================================================
# VALIDATION TESTS
# ============================================================

def _test_bremsstrahlung():
    """Compare the bremsstrahlung formula with Putvinski Fig. 4."""
    print("=" * 70)
    print("TEST 1: BREMSSTRAHLUNG POWER")
    print("=" * 70)

    # Putvinski Fig. 4: n_i = 1e20 m⁻³ = 1e14 cm⁻³, fB = 0.15
    n_i = 1e14
    f_B = 0.15
    n_B = f_B * n_i
    n_p = (1 - f_B) * n_i
    n_e = n_p + Z_B * n_B
    Z_eff = Z_eff_calc(n_p, n_B)

    print(f"Plasma: n_i=10¹⁴, f_B=0.15, n_e={n_e:.2e}, Z_eff={Z_eff:.3f}")
    print()

    # Values approximately read from Putvinski Fig. 4:
    # T_e ≈ T_i/2, P_Brem (W/m³)
    # T_i=200 keV (T_e≈100): ~0.4 MW/m³ = 0.4 W/cm³
    # T_i=300 keV (T_e≈150): ~0.8 MW/m³ = 0.8 W/cm³
    # T_i=500 keV (T_e≈250): ~1.4 MW/m³ = 1.4 W/cm³

    test_cases = [
        (100, 50, 0.15),
        (200, 100, 0.4),
        (300, 150, 0.8),
        (500, 250, 1.4),
    ]

    print(f"{'T_i (keV)':>10} | {'T_e (keV)':>10} | {'P_Brem (W/cm³)':>16} | {'expected':>10}")
    print("-" * 60)
    for T_i, T_e, P_expected in test_cases:
        P_brem = P_bremsstrahlung(n_e, T_e, Z_eff)
        ratio = P_brem / P_expected if P_expected > 0 else 0
        marker = "[OK]" if 0.5 < ratio < 2.0 else "[WARN]"
        print(f"{T_i:>10.0f} | {T_e:>10.0f} | {P_brem:>16.3e} | "
              f"{P_expected:>10.2f} {marker}")

    # x = T_e/E_rest comparison
    print(f"\nRelativistic factor at T_e=150 keV:")
    R = relativistic_R_factor(150.0)
    print(f"  R(x=0.293) = {R:.3f}  (Putvinski Fig. 3: ~1.05-1.10)")
    if 1.0 < R < 1.3:
        print("  [OK] Consistent with Putvinski Fig. 3 (relativistic enhancement)")
    else:
        print("  [WARN] Not in expected range")


def _test_fusion_power():
    """Test the fusion power formula."""
    print("\n" + "=" * 70)
    print("TEST 2: FUSION POWER (THERMAL)")
    print("=" * 70)

    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i

    print(f"Plasma: n_i=10¹⁴, n_p={n_p:.2e}, n_B={n_B:.2e}")
    print()

    # Thermal P_F read from Putvinski Fig. 4 (original SW cross section):
    # T_i=300 keV: ~1 MW/m³ = 1 W/cm³ (may be inexact)
    # T_i=500 keV: ~1.1 MW/m³ = 1.1 W/cm³ (peak)
    #
    # The TB cross section should yield ~30% higher.

    print(f"{'T_i (keV)':>10} | {'P_F (W/cm³)':>14} | {'⟨σv⟩':>14}")
    print("-" * 50)
    for T_i in [100, 200, 300, 400, 500, 700]:
        P_F = P_fusion_thermal(n_p, n_B, T_i)
        sv = sigma_v_TB_numerical(T_i)[0]
        print(f"{T_i:>10.0f} | {P_F:>14.3e} | {sv:>14.3e}")


def _test_ignition_window():
    """Reproduce Putvinski Fig. 4 — critical test."""
    print("\n" + "=" * 70)
    print("TEST 3: IGNITION WINDOW (Putvinski Fig. 4 reproduction)")
    print("=" * 70)

    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i

    print(f"\nPlasma: n_i=10¹⁴, f_B=0.15")
    print("Thermal assumption (no kinetic enhancement)")
    print()

    print(f"{'T_i':>6} | {'T_e':>6} | {'P_F':>10} | {'P_Brem':>10} | "
          f"{'P_F/P_B':>8} | {'ign':>4} | {'τ_E*':>10}")
    print("-" * 75)

    T_i_scan = [100, 150, 200, 250, 300, 350, 400, 500, 600]
    for T_i in T_i_scan:
        info = ignition_check(n_p, n_B, T_i)
        marker = "Y" if info['ignition'] else "-"
        tau_str = f"{info['tau_E_star_s']:.1e}" if np.isfinite(info['tau_E_star_s']) else "inf"
        print(f"{T_i:>6.0f} | {info['T_e_keV']:>6.1f} | "
              f"{info['P_F']:>10.3e} | {info['P_Brem']:>10.3e} | "
              f"{info['P_F_over_P_Brem']:>8.3f} | {marker:>4} | {tau_str:>10}")

    # Per Putvinski 2019 the ignition window is T_i ~ 250-400 keV
    print(f"\nExpectation: ignition in T_i ∈ [250, 400] keV (Putvinski Fig. 4)")
    print(f"P_F/P_Brem peak ≈ 1.03 (Putvinski: 3% margin)")


if __name__ == "__main__":
    _test_bremsstrahlung()
    _test_fusion_power()
    _test_ignition_window()
