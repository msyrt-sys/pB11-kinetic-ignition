"""
α-p elastik saçılma kesiti - R-matrix faz kayma yaklaşımı.

KAYNAKLAR:
- Hale, G.M. (1990) - LA-UR-90-3756, R-matrix analysis of p-4He system
- Plattner-Bacher (1971), Phys. Rev. C 3, 2100 - Phase shift analysis
- Brandan-Plattner-Haeberli (1976), Nucl. Phys. A 263, 189 - Effective range
- Hupin et al. (2014), arXiv:1409.0892 - Ab initio NCSMC

Yaklaşım: Coulomb + nükleer faz kaymalarından σ(E_p_lab, θ_lab) hesaplama.

⁵Li enerji seviyeleri (proton CM enerji cinsinden):
- Ground state: 3/2- @ E_p_CM = 1.97 MeV (lab E_p ≈ 9.85 MeV proton üzerine 4He)
  Ama biz inverse kinematics yapıyoruz: alpha üzerine proton.
  Standardize edelim: E_alpha_CM = E_alpha * m_p/(m_a+m_p) = E_alpha/5
  ⁵Li resonance E_α_lab ≈ 9.85 MeV (proton üstünde 4He), 
  bizim inverse: E_α_lab × 4 ≈ 1.97 MeV (4He üstünde p)
  Karşılık: bizim alpha üstünde proton sistemde:
  E_p_lab_proton = (m_a+m_p)/m_p × E_p_CM = 5 × 1.97 = 9.85 MeV proton
  Inverse kinematics: E_alpha_lab = (m_a+m_p)/m_a × E_p_CM = 5/4 × 1.97 = 2.46 MeV

  Ama BIZIM grafiklerimizde resonance ~8 MeV alpha lab.
  Bu 8/5 = 1.6 MeV CM = E_p_CM(if proton beam) = 4 × 1.6 = 6.4 MeV proton lab
  Bu farklı bir bölge — muhtemelen P-wave 1/2- excited state ⁵Li (E* ~ 4 MeV)

Sayısal hesap için:
σ(θ) = |f_C(θ) + f_N(θ)|²
f_C: Coulomb amplitude (Rutherford'tan)
f_N: nuclear amplitude (faz kaymalardan)

Pragmatic fit: We will fit a DIRECT FUNCTION σ_s/σ_R(E, θ) that reproduces 
Belloni 2021 Fig. 2 contour structure.
"""

import numpy as np
# scipy.special not needed for this implementation

# Kinematik sabitler  
m_a_amu = 4.002602  # exact 4He mass
m_p_amu = 1.007276
amu_MeV = 931.494
m_a_MeV = m_a_amu * amu_MeV
m_p_MeV = m_p_amu * amu_MeV
mu_MeV = m_a_MeV * m_p_MeV / (m_a_MeV + m_p_MeV)  # reduced mass
e2_MeVfm = 1.43997  # alpha hbar c
hbar_c_MeVfm = 197.327


def coulomb_eta(E_CM_MeV, Z1=1, Z2=2):
    """Coulomb (Sommerfeld) parameter η."""
    v_rel = np.sqrt(2 * E_CM_MeV / mu_MeV)  # natural units (c=1)
    eta = Z1 * Z2 * e2_MeVfm / (hbar_c_MeVfm * v_rel)
    return eta


def coulomb_amplitude(E_CM_MeV, theta_CM_rad, Z1=1, Z2=2):
    """Coulomb scattering amplitude f_C(θ) (CM frame).
    
    f_C(θ) = -η/(2k sin²(θ/2)) × exp(-iη ln(sin²(θ/2)) + 2i σ_0)
    
    Modulus squared verir Rutherford'u.
    """
    v_rel = np.sqrt(2 * E_CM_MeV / mu_MeV)
    k = mu_MeV * v_rel / hbar_c_MeVfm  # 1/fm
    eta = coulomb_eta(E_CM_MeV, Z1, Z2)
    
    sin2 = np.sin(theta_CM_rad / 2)**2
    f_C_mag = eta / (2 * k * sin2)
    # Coulomb phase: exp(-iη ln(sin²(θ/2)))
    f_C_phase = -eta * np.log(sin2)
    
    f_C = -f_C_mag * np.exp(1j * f_C_phase)
    return f_C


def nuclear_amplitude_pragmatic(E_alpha_lab_MeV, theta_alpha_CM_rad):
    """Pragmatik nuclear amplitude — Belloni'nin σ_s/σ_R yapısını reproduce eder.
    
    ⁵Li yapısının ana özellikleri:
    - Ground state 3/2- @ E_α_lab ≈ 2.5 MeV (zayıf)
    - 1/2- excited state @ E_α_lab ≈ 8 MeV (Belloni'nin gördüğü resonance)
    
    Bu fonksiyon yaklaşık f_N hesaplar, sadece l=0,1 dalgaları için.
    Belloni Fig. 2'yi yaklaşık reproduce eder.
    """
    E_CM = E_alpha_lab_MeV / 5  # m_p/(m_a+m_p) = 1/5 with mass=4
    
    v_rel = np.sqrt(2 * E_CM / mu_MeV)
    k = mu_MeV * v_rel / hbar_c_MeVfm
    
    # ⁵Li 3/2- resonance (P-wave dominant)
    E_R1 = 1.6  # MeV CM (≈ 8 MeV alpha lab) — P3/2 strong resonance
    Gamma1 = 1.5  # MeV width
    
    # ⁵Li 1/2- (lower energy P-wave, weaker)
    E_R2 = 5.0  # MeV CM
    Gamma2 = 5.0  # MeV
    
    # Resonance amplitudes (Breit-Wigner):
    # f_l = (1/k) × Σ (2l+1) × [exp(2iδ_l) - 1] / 2i × P_l(cos θ)
    # 
    # For a single resonance: tan(δ_l) = (Γ/2)/(E_R - E)
    
    # P-wave (l=1)
    # Breit-Wigner phase shift
    delta_p = np.arctan2(Gamma1/2, E_R1 - E_CM)
    
    # P_1(cos θ) = cos θ
    cos_th = np.cos(theta_alpha_CM_rad)
    P1 = cos_th
    
    # f_N P-wave only (ana terim)
    # f_l = ((2l+1)/k) × exp(iδ_l) × sin(δ_l) × P_l(cos θ)
    f_N_P = (3/k) * np.exp(1j * delta_p) * np.sin(delta_p) * P1
    
    # S-wave (l=0) - genelde zayıf p-α'da
    # Effective range: tan(δ_0) ~ -k×a (a=scattering length)
    a_S = -1.5  # fm (typical for p-4He s-wave)
    delta_s = np.arctan(-k * a_S)
    f_N_S = (1/k) * np.exp(1j * delta_s) * np.sin(delta_s)
    
    # D-wave - high energy contribution
    # Coupling to 5Li 5/2- @ E_α ~ 16 MeV
    E_R_D = 3.0  # MeV CM (~15 MeV alpha lab)
    Gamma_D = 4.0
    delta_d = np.arctan2(Gamma_D/2, E_R_D - E_CM)
    P2 = (3 * cos_th**2 - 1) / 2
    f_N_D = (5/k) * np.exp(1j * delta_d) * np.sin(delta_d) * P2
    
    return f_N_S + f_N_P + f_N_D


def sigma_s_phase_shift(E_alpha_lab_MeV, theta_p_lab_rad):
    """σ_s(E_α, θ_p,lab) - phase shift hesabı.
    
    1. theta_p_lab → theta_alpha_CM (kinematic)
    2. f_C + f_N total amplitude
    3. |f|² = differential cross section CM
    4. Lab frame'e çevir (Jacobian)
    """
    # CM angle for alpha (recoil proton lab → alpha CM)
    # θ_p,lab = (π - θ_α,CM)/2 → θ_α,CM = π - 2θ_p,lab
    theta_alpha_CM = np.pi - 2 * theta_p_lab_rad
    
    E_CM = E_alpha_lab_MeV / 5
    
    f_C = coulomb_amplitude(E_CM, theta_alpha_CM)
    f_N = nuclear_amplitude_pragmatic(E_alpha_lab_MeV, theta_alpha_CM)
    
    # Total cross section CM
    sigma_CM = np.abs(f_C + f_N)**2  # fm²/sr
    
    # Lab frame Jacobian for proton recoil:
    # dΩ_CM_alpha / dΩ_lab_proton = 4 cos(θ_p,lab)
    sigma_lab = sigma_CM * 4 * np.cos(theta_p_lab_rad)
    
    return sigma_lab * 10  # fm²/sr → mb/sr


def rutherford_lab(E_alpha_lab_MeV, theta_p_lab_rad):
    """Saf Rutherford lab frame proton recoil."""
    E_CM = E_alpha_lab_MeV / 5
    theta_alpha_CM = np.pi - 2 * theta_p_lab_rad
    
    # CM Rutherford
    sigma_R_CM = (1 * 2 * e2_MeVfm)**2 / (4 * E_CM)**2 / np.sin(theta_alpha_CM/2)**4
    sigma_R_lab = sigma_R_CM * 4 * np.cos(theta_p_lab_rad)
    return sigma_R_lab * 10


# Test: Compute σ_s and σ_R at various E and θ
print("Phase-shift tabanlı σ_s/σ_R kontrol:")
print(f"{'E_α (MeV)':>10} | {'θ_lab':>6} | {'σ_s (mb/sr)':>14} | {'σ_R (mb/sr)':>14} | {'ratio':>6}")
print("-" * 70)

for E_alpha in [0.4, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
    for theta_deg in [1, 30, 60, 75, 85]:
        theta_rad = np.radians(theta_deg)
        sig_s = sigma_s_phase_shift(E_alpha, theta_rad)
        sig_R = rutherford_lab(E_alpha, theta_rad)
        ratio = sig_s / sig_R if sig_R > 0 else 0
        print(f"{E_alpha:>10.1f} | {theta_deg:>5}° | {sig_s:>14.1f} | {sig_R:>14.1f} | {ratio:>6.2f}")
    print()
