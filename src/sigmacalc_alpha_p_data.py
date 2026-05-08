"""
sigmacalc_alpha_p_data.py — SigmaCalc 2.0 evaluated α-p elastic cross sections

Source: SigmaCalc 2.0 (Gurbich, A.F., Nucl. Instr. Meth. B 371, 27-32, 2016)
URL: http://sigmacalc.obninsk.ru/
Date retrieved: 2 May 2026
Reaction: 1H(α,p)4He elastic scattering (inverse kinematics)

Data extracted manually from plot images at 10 angles (lab frame, proton recoil):
1°, 5°, 10°, 20°, 30°, 60°, 70°, 75°, 80°, 85°

CRITICAL: Plot Y-axis labels "E+N" indicate scale factor 10^N.
Example: "E+4" with reading "2.40" means σ = 2.40 × 10^4 = 24000 mb/sr.

Energy range: 400-12400 keV (lab frame, alpha incident)
Cross section: mb/sr (lab frame)

Validation: 400 keV readings match Rutherford within 5-10% at all angles
(low-energy regime is Coulomb-dominated, as expected).

Citation:
- A.F. Gurbich, Nucl. Instr. Meth. B 371, 27-32 (2016) - SigmaCalc 2.0
- F. Belloni, Plasma Phys. Control. Fusion 63, 055020 (2021)
"""

import numpy as np

# Energy grid (keV, lab frame)
E_alpha_keV = np.array([
    400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 
    3600, 4000, 4400, 4800, 5600, 6400, 7200,
    8000, 8400, 9200, 10000, 10800, 11600, 12400
])

# Lab-frame proton recoil angles (degrees)
theta_p_lab_deg = np.array([1, 5, 10, 20, 30, 60, 70, 75, 80, 85])

# Differential cross section sigma_s in mb/sr
# CORRECTED: All values in actual mb/sr (scale factor applied)

sigma_s_mb_per_sr = np.array([
    # 1° (E+3 scale)
    [3.20e3, 1.50e3, 0.80e3, 0.55e3, 0.45e3, 0.40e3, 0.40e3, 0.42e3,
     0.50e3, 0.65e3, 0.85e3, 1.10e3, 1.50e3, 1.85e3, 1.95e3,
     2.00e3, 2.00e3, 1.90e3, 1.75e3, 1.60e3, 1.50e3, 1.40e3],
    # 5° (E+3)
    [3.10e3, 1.45e3, 0.78e3, 0.52e3, 0.42e3, 0.37e3, 0.38e3, 0.40e3,
     0.50e3, 0.65e3, 0.85e3, 1.10e3, 1.50e3, 1.85e3, 1.95e3,
     2.00e3, 2.00e3, 1.90e3, 1.75e3, 1.60e3, 1.45e3, 1.30e3],
    # 10° (E+3)
    [3.20e3, 1.50e3, 0.80e3, 0.55e3, 0.42e3, 0.34e3, 0.32e3, 0.33e3,
     0.40e3, 0.55e3, 0.75e3, 1.00e3, 1.40e3, 1.70e3, 1.80e3,
     1.85e3, 1.85e3, 1.75e3, 1.60e3, 1.45e3, 1.30e3, 1.20e3],
    # 20° (E+3)
    [3.60e3, 1.85e3, 1.05e3, 0.65e3, 0.45e3, 0.32e3, 0.27e3, 0.27e3,
     0.32e3, 0.45e3, 0.60e3, 0.80e3, 1.05e3, 1.15e3, 1.20e3,
     1.20e3, 1.20e3, 1.15e3, 1.05e3, 0.95e3, 0.85e3, 0.80e3],
    # 30° (E+3)
    [4.80e3, 2.50e3, 1.40e3, 0.85e3, 0.55e3, 0.40e3, 0.32e3, 0.30e3,
     0.30e3, 0.35e3, 0.42e3, 0.50e3, 0.65e3, 0.78e3, 0.80e3,
     0.80e3, 0.80e3, 0.78e3, 0.72e3, 0.68e3, 0.65e3, 0.60e3],
    # 60° (E+4)
    [2.40e4, 0.90e4, 0.45e4, 0.25e4, 0.15e4, 0.09e4, 0.055e4, 0.030e4,
     0.018e4, 0.010e4, 0.006e4, 0.004e4, 0.004e4, 0.005e4, 0.005e4,
     0.005e4, 0.005e4, 0.005e4, 0.004e4, 0.004e4, 0.003e4, 0.003e4],
    # 70° (E+4) - new data
    [7.6e4, 2.5e4, 1.0e4, 0.50e4, 0.30e4, 0.20e4, 0.15e4, 0.12e4,
     0.10e4, 0.09e4, 0.08e4, 0.08e4, 0.08e4, 0.08e4, 0.08e4,
     0.08e4, 0.08e4, 0.08e4, 0.08e4, 0.08e4, 0.077e4, 0.076e4],
    # 75° (E+5)
    [1.75e5, 0.50e5, 0.20e5, 0.10e5, 0.060e5, 0.040e5, 0.030e5, 0.025e5,
     0.020e5, 0.018e5, 0.016e5, 0.015e5, 0.014e5, 0.013e5, 0.012e5,
     0.012e5, 0.012e5, 0.011e5, 0.010e5, 0.010e5, 0.009e5, 0.008e5],
    # 80° (E+5)
    [6.0e5, 1.5e5, 0.6e5, 0.30e5, 0.18e5, 0.12e5, 0.090e5, 0.075e5,
     0.060e5, 0.050e5, 0.045e5, 0.040e5, 0.038e5, 0.035e5, 0.033e5,
     0.032e5, 0.032e5, 0.030e5, 0.028e5, 0.025e5, 0.023e5, 0.020e5],
    # 85° (E+6)
    [4.8e6, 1.2e6, 0.50e6, 0.25e6, 0.15e6, 0.10e6, 0.075e6, 0.060e6,
     0.050e6, 0.045e6, 0.040e6, 0.035e6, 0.030e6, 0.028e6, 0.025e6,
     0.024e6, 0.024e6, 0.022e6, 0.020e6, 0.018e6, 0.017e6, 0.015e6],
])


def sigma_s_alpha_p(E_alpha_keV_query, theta_lab_deg_query):
    """SigmaCalc evaluated σ_s(E_α, θ_lab). Linear interp. Returns mb/sr."""
    from scipy.interpolate import RegularGridInterpolator
    
    interp = RegularGridInterpolator(
        (theta_p_lab_deg, E_alpha_keV),
        sigma_s_mb_per_sr,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    pts = np.column_stack([
        np.atleast_1d(theta_lab_deg_query),
        np.atleast_1d(E_alpha_keV_query)
    ])
    return interp(pts)


def rutherford_inverse_kinematics_lab(E_alpha_keV_query, theta_lab_deg_query):
    """Lab-frame Rutherford σ_R for proton recoil. Returns mb/sr."""
    e2_MeVfm = 1.44
    Z_a, Z_p = 2, 1
    m_a, m_p = 4, 1
    
    E_a_MeV = np.atleast_1d(E_alpha_keV_query) / 1000
    th_lab = np.radians(theta_lab_deg_query)
    
    E_CM = E_a_MeV * m_p / (m_a + m_p)
    th_CM = np.pi - 2 * th_lab
    
    sigma_R_CM = (Z_a * Z_p * e2_MeVfm)**2 / (4 * E_CM)**2 / np.sin(th_CM/2)**4
    sigma_R_lab = sigma_R_CM * 4 * np.cos(th_lab)
    
    return sigma_R_lab * 10  # fm^2/sr → mb/sr


def dsigma_dEp_at_Ealpha(E_alpha_MeV, n_theta_points=200):
    """dσ/dE_p kernel at fixed E_α, integrating over angles.
    
    Returns:
        E_p_arr (MeV, sorted ascending), dsigma_dEp_arr (mb/MeV)
    """
    # Cover available range: 1° to 85° (data limit)
    theta_grid_deg = np.linspace(1, 85, n_theta_points)
    theta_grid = np.radians(theta_grid_deg)
    
    E_alpha_keV_val = E_alpha_MeV * 1000
    sigma_s_arr = np.array([sigma_s_alpha_p(E_alpha_keV_val, td)[0] 
                             for td in theta_grid_deg])
    
    E_p_arr_MeV = 0.64 * E_alpha_MeV * np.cos(theta_grid)**2
    dsig_dEp = sigma_s_arr * np.pi / (0.64 * E_alpha_MeV * np.cos(theta_grid))
    
    idx = np.argsort(E_p_arr_MeV)
    return E_p_arr_MeV[idx], dsig_dEp[idx]


def sigma_s_total_forward(E_alpha_keV_query):
    """Forward-hemisphere total integrated σ_s in mb."""
    E_query = np.atleast_1d(E_alpha_keV_query)
    sigma_total = np.zeros_like(E_query, dtype=float)
    
    theta_grid = np.radians(theta_p_lab_deg)
    
    for k, E in enumerate(E_query):
        sigma_at_E = np.zeros(len(theta_p_lab_deg))
        for i, theta_deg in enumerate(theta_p_lab_deg):
            sigma_at_E[i] = sigma_s_alpha_p(E, theta_deg)[0]
        integrand = sigma_at_E * np.sin(theta_grid)
        sigma_total[k] = 2 * np.pi * np.trapezoid(integrand, theta_grid)
    
    return sigma_total


# Validation script
if __name__ == "__main__":
    print("SigmaCalc α-p elastic data:")
    print(f"  Angles: {theta_p_lab_deg}")
    print(f"  Energy range: {E_alpha_keV[0]}-{E_alpha_keV[-1]} keV")
    print(f"  Total data points: {len(E_alpha_keV) * len(theta_p_lab_deg)}")
    print()
    
    print("σ_s vs σ_R at 400 keV (Coulomb-dominated, ratio ≈ 1):")
    print(f"{'θ':>4} | {'σ_s (mb/sr)':>14} | {'σ_R (mb/sr)':>14} | {'ratio':>6}")
    for theta in theta_p_lab_deg:
        sig_s = sigma_s_alpha_p(400, theta)[0]
        sig_R = rutherford_inverse_kinematics_lab(400, theta)[0]
        ratio = sig_s / sig_R if sig_R > 0 else 0
        print(f"{theta:>3}° | {sig_s:>14.0f} | {sig_R:>14.0f} | {ratio:>6.2f}")
    
    print()
    print("σ_s vs σ_R at 8000 keV (⁵Li resonance, nuclear effects):")
    for theta in theta_p_lab_deg:
        sig_s = sigma_s_alpha_p(8000, theta)[0]
        sig_R = rutherford_inverse_kinematics_lab(8000, theta)[0]
        ratio = sig_s / sig_R if sig_R > 0 else 0
        print(f"{theta:>3}° | {sig_s:>14.1f} | {sig_R:>14.1f} | {ratio:>6.2f}")
    
    print()
    print("Energy transfer kernel test (E_α = 4 MeV):")
    E_p, dsig = dsigma_dEp_at_Ealpha(4.0)
    sigma_total = np.trapezoid(dsig, E_p)
    print(f"  Total σ from kernel: {sigma_total:.1f} mb")
    print(f"  Range E_p: {E_p[0]*1000:.1f} - {E_p[-1]*1000:.1f} keV")
