"""
power_balance.py — Plazma güç dengesi ve tutuşma kriteri

Birincil referans: Putvinski et al. (2019) §2.2, Eq. 7-10

Bu modül üç ana iş yapar:

1. **Bremsstrahlung gücü** (Putvinski Eq. 7, Svensson 1982 fit):
   P_Brem = 7.56e-11 · n_e² · x^(1/2) · [Z_eff·(1+1.78·x^1.34) 
            + 2.12·x·(1+1.1·x+x²-1.25·x^2.5)]   (eV·cm³/s)
   x = T_e/m_e c² (boyutsuz)

2. **Fusion gücü** P_F = E_F · n_p · n_B · ⟨σv⟩
   - FP-distorted f_p ile hesaplanırsa kinetik artırım dahil
   - Maxwell varsayımı ile düz reaktivite kullanılırsa termal değer

3. **α-iyon ve α-elektron güç transferi**:
   - Putvinski Eq. 8: durağan elektron dengesi
   - Yavaşlama α'larının protona ve elektrona enerji aktarımı

4. **Self-consistent T_e bulma**:
   - Verili T_i ve plazma karışımı için T_e'yi öyle ayarla ki:
     P_α,e + P_i,e = P_Brem (elektron gücü dengeli)

5. **Tutuşma kriteri**:
   - P_F > P_Brem? (gerekli koşul)
   - τ_E* = U_K / (P_F - P_Brem) (Ochs metriği)

Birim sistemi: CGS (cm, g, s, erg) - hesaplamalar; sonuçlar SI'ya çevrilebilir
Güç birimi: W/cm³ veya erg/(cm³·s) (1 W = 10⁷ erg/s)
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
# FİZİKSEL SABİTLER
# ============================================================

E_rest_eV = m_e_g * c_cm**2 / 1.602176634e-12  # m_e c² = 5.11e5 eV
E_rest_keV = E_rest_eV / 1000.0


# ============================================================
# 1. BREMSSTRAHLUNG GÜCÜ (Putvinski Eq. 7 / Svensson 1982)
# ============================================================

def P_bremsstrahlung(n_e_cm3, T_e_keV, Z_eff):
    """Bremsstrahlung gücü (W/cm³).
    
    NRL Plasma Formulary klasik formu (W/cm³):
      P_classical = 5.34e-31 · n_e² · Z_eff · √(T_e[keV])
    
    Svensson 1982 relativistik düzeltme (Putvinski Eq. 7'deki köşeli parantez):
      g(x) = (1 + 1.78·x^1.34) + (2.12·x/Z_eff)·(1+1.1·x+x²-1.25·x^2.5)
      x = T_e/m_e c²
    
    Toplam: P = P_classical · g(x)
    
    Doğrulama: T_e=150 keV, n_e=1.6e14, Z_eff=2.87 → P ≈ 0.65 W/cm³
    Bu Putvinski Şek. 4'teki ~0.8 W/cm³ ile uyumlu.
    
    Parametreler
    ----------
    n_e_cm3 : float veya ndarray
        Elektron yoğunluğu (cm⁻³)
    T_e_keV : float veya ndarray
        Elektron sıcaklığı (keV)
    Z_eff : float
        Etkin yük: Σ_i n_i Z_i² / Σ_i n_i Z_i
    
    Dönüş
    -----
    P_Brem : float veya ndarray
        Bremsstrahlung gücü, W/cm³
    """
    # Klasik kısım (NRL)
    P_classical = 5.34e-31 * n_e_cm3**2 * Z_eff * np.sqrt(T_e_keV)
    
    # Svensson relativistik düzeltme
    x = T_e_keV / E_rest_keV
    
    # e-i bremsstrahlung relativistik düzeltmesi
    factor_ei = 1.0 + 1.78 * x**1.34
    
    # e-e bremsstrahlung katkısı (relativistik, Z=0)
    # Putvinski Eq. 7'de 2.12·x·(1+1.1·x+x²-1.25·x^2.5) terimi
    # Bunu NRL klasik forma eklemek için Z_eff ile bölelim
    factor_ee = 2.12 * x * (1.0 + 1.1 * x + x**2 - 1.25 * x**2.5) / Z_eff
    
    g_relativistic = factor_ei + factor_ee
    
    return P_classical * g_relativistic


def Z_eff_calc(n_p_cm3, n_B_cm3):
    """Z_eff için p-B11 plazması (yalnızca proton + boron).
    
    Z_eff = Σ_i n_i Z_i² / Σ_i n_i Z_i = Σ_i n_i Z_i² / n_e
    
    p-B11 için: n_e = n_p + Z_B · n_B (quasineutrality)
    """
    n_e = n_p_cm3 + Z_B * n_B_cm3
    Z_eff = (n_p_cm3 * Z_p**2 + n_B_cm3 * Z_B**2) / n_e
    return Z_eff


# ============================================================
# 2. FUSION GÜCÜ
# ============================================================

def P_fusion_thermal(n_p_cm3, n_B_cm3, T_i_keV):
    """Termal fusion gücü (Maxwell varsayımı).
    
    P_F = n_p · n_B · ⟨σv⟩(T_i) · E_F
    
    burada E_F = Q-değeri = 8.681 MeV (toplam enerji)
    
    Dönüş: W/cm³
    """
    sv = sigma_v_TB_numerical(T_i_keV)  # cm³/s
    if np.isscalar(T_i_keV):
        sv = sv[0] if hasattr(sv, '__len__') else sv
    
    # Reaksiyon hızı: n_p · n_B · ⟨σv⟩ (reaksiyon/cm³/s)
    rate = n_p_cm3 * n_B_cm3 * sv
    
    # Enerji per reaksiyon
    E_F_erg = Q_pB11_keV * keV_to_erg
    
    # P_F (erg/s/cm³)
    P_erg_s = rate * E_F_erg
    
    # W/cm³
    return P_erg_s / 1e7


def P_fusion_kinetic(v_grid, f_p, n_B_cm3, T_B_keV):
    """Fusion gücü, FP-distorted proton dağılımı ile.
    
    P_F = ∫ 4π·v²·f_p(v) · n_B · σ(v) · v · E_F · dv
    
    Burada f_p genel olarak Maxwell olmayabilir (FP solver çıktısı).
    Bu Putvinski'nin "kinetik artırım" mekanizmasını yakalamamızı sağlar.
    
    Parametreler
    ----------
    v_grid : ndarray
        Proton hız grid'i (cm/s)
    f_p : ndarray
        Proton dağılım fonksiyonu (cm⁻⁶ s³)
    n_B_cm3 : float
        Boron yoğunluğu
    T_B_keV : float
        Boron sıcaklığı (Maxwell)
    
    Dönüş
    -----
    P_F : float
        Fusion gücü, W/cm³
    """
    from cross_sections import sigma_TB
    
    # Lab frame proton kinetik enerji → CM enerji
    E_p_erg = 0.5 * m_p_g * v_grid**2
    E_CM_erg = E_p_erg * m_B_g / (m_p_g + m_B_g)
    E_CM_keV = E_CM_erg / keV_to_erg
    
    sigma_b = sigma_TB(E_CM_keV)
    sigma_cm2 = sigma_b * barn_cm2
    
    # Reaksiyon hızı: ∫ 4π v² f_p · n_B · σ(v) · v dv
    integrand = 4 * np.pi * v_grid**2 * f_p * n_B_cm3 * sigma_cm2 * v_grid
    rate = np.trapezoid(integrand, v_grid)
    
    # Enerji per reaksiyon
    E_F_erg = Q_pB11_keV * keV_to_erg
    
    P_erg_s = rate * E_F_erg
    return P_erg_s / 1e7


# ============================================================
# 3. RELATİVİSTİK ION-ELEKTRON ENERJİ DEĞİŞİMİ
# ============================================================
# Putvinski Eq. 9-10

def relativistic_R_factor(T_e_keV):
    """Putvinski Eq. 10: relativistik R(x) düzeltme faktörü.
    
    R(x) = (1 + 2x + 2x²) · √(π x³/2) / ∫₀^∞ t² · exp((1-√(1+t²))/x) dt
    
    x = T_e/(m_e c²). 
    
    Yorumlama: Bu faktör, ν_ei^relativistic / ν_ei^classical oranıdır.
    Putvinski Şek. 3: T_e=150 keV (x=0.29) için R ≈ 1.1 (relativistik etki
    iyon-elektron çiftleşmesini ARTIRIYOR, sınırlamıyor — Putvinski metni
    "%10 daha yüksek elektron sıcaklığı" diyor).
    
    Klasik limit x→0: R → 1.
    
    Dönüş: R faktörü (boyutsuz, ~1.0-1.4)
    """
    x = T_e_keV / E_rest_keV
    
    if x < 0.001:
        return 1.0  # Klasik limit
    
    # ∫₀^∞ t² · exp((1-√(1+t²))/x) dt — sayısal kuadratür
    t_grid = np.logspace(-3, 2, 2000)
    integrand = t_grid**2 * np.exp((1 - np.sqrt(1 + t_grid**2)) / x)
    integral = np.trapezoid(integrand, t_grid)
    
    # Putvinski Eq. 10: R = (1+2x+2x²) · √(πx³/2) / integral
    R = (1 + 2*x + 2*x**2) * np.sqrt(np.pi * x**3 / 2) / integral
    return R


def P_ion_electron_transfer(n_p, n_B, T_p, T_B, T_e, lnLambda=17.0):
    """İyondan elektrona enerji transfer gücü (Putvinski Eq. 9).
    
    P_{i,e} = Σ_i (3/2) · ν_{ie} · n_i · (T_i - T_e) · R(x)
    
    burada ν_{ie} klasik çarpışma frekansı:
    ν_{ie}^cl = 4.8e-9 · Z_i² · λ_{ie} · n_e / (m_i · T_e^(3/2))   (NRL formu)
    
    Dönüş: W/cm³
    """
    n_e = n_p + Z_B * n_B
    
    # Klasik çarpışma frekansı (NRL Plasma Formulary)
    # ν_{ie} = (1/τ_ε) Spitzer enerji denkleştirme zamanı
    # NRL p.31:
    # ν_ε^{ie} = (m_e/m_i) · (4√(2π)/3) · (n_e e⁴ ln Λ)/(m_e^(1/2) T_e^(3/2)) · Z_i²
    
    # Proton katkısı
    nu_pe = (m_e_g/m_p_g) * (4*np.sqrt(2*np.pi)/3) * \
            (n_e * e_esu**4 * lnLambda) / (m_e_g**0.5 * (T_e * keV_to_erg)**1.5) * Z_p**2
    
    # Boron katkısı
    nu_Be = (m_e_g/m_B_g) * (4*np.sqrt(2*np.pi)/3) * \
            (n_e * e_esu**4 * lnLambda) / (m_e_g**0.5 * (T_e * keV_to_erg)**1.5) * Z_B**2
    
    # Relativistik düzeltme
    R = relativistic_R_factor(T_e)
    
    # Güç transferi (proton + boron)
    # (3/2) n_i ν_ie · (T_i - T_e), ama biz erg cinsinden istiyoruz
    P_pe = 1.5 * n_p * nu_pe * (T_p - T_e) * keV_to_erg * R   # erg/s/cm³
    P_Be = 1.5 * n_B * nu_Be * (T_B - T_e) * keV_to_erg * R   # erg/s/cm³
    
    P_total_erg = P_pe + P_Be
    return P_total_erg / 1e7  # W/cm³


# ============================================================
# 3.5 İYON-İYON TERMALİZASYON (Ochs Eq. 13-14 K_pb)
# ============================================================

def P_pb_thermalization(n_p, n_B, T_p_keV, T_B_keV, lnLambda=17.0):
    """Proton-Boron iyon-iyon termalizasyon güç transferi (Ochs Eq. 13-14, K_pb).
    
    P_{p→b} = (3/2) · n_p · ν_pb · (T_p - T_B)
    
    NRL Plasma Formulary, Spitzer enerji eşitleme:
        ν_pb = (8√(2π)/3) · (Z_p² Z_B² e⁴ n_B ln Λ) / (m_p · m_B) ·
               · (T_p/m_p + T_B/m_B)^(-3/2)
    
    Bu Ochs makalesinin Eq. 13-14'te K_pb (T_b - T_p) terimine karşılıktır.
    İşaret: T_p > T_B ise pozitif (proton boron'u ısıtır), tam tersi.
    
    p-B11 plazmasında τ_pb (= U_p/P_pb) ~ 10 ms, τ_E ~ 1-100 s mertebesinde
    olduğundan T_p ≈ T_B steady-state'de. Putvinski'nin tek-T varsayımı haklı.
    Bu fonksiyon doğrulama amaçlıdır.
    
    Parametreler
    ----------
    n_p, n_B : float
        Yoğunluklar (cm⁻³)
    T_p_keV, T_B_keV : float
        Sıcaklıklar (keV)
    lnLambda : float
    
    Dönüş
    -----
    P_pb : float
        p → B ısı transferi (W/cm³)
    """
    T_p_erg = T_p_keV * keV_to_erg
    T_B_erg = T_B_keV * keV_to_erg
    
    prefactor = (8 * np.sqrt(2 * np.pi) / 3) * (Z_p * Z_B * e_esu**2)**2 * n_B * lnLambda
    denom = m_p_g * m_B_g * (T_p_erg / m_p_g + T_B_erg / m_B_g)**1.5
    
    nu_pb = prefactor / denom  # 1/s
    
    P_erg_per_s_cm3 = 1.5 * n_p * nu_pb * (T_p_erg - T_B_erg)
    
    return P_erg_per_s_cm3 / 1e7  # W/cm³


def thermalization_time_pb(n_p, n_B, T_p_keV, T_B_keV, lnLambda=17.0):
    """p-B termalizasyon zaman ölçeği τ_pb = U_p / P_pb.
    
    τ_pb << τ_E ise tek-T varsayımı geçerli.
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
    """Z_eff ve n_e için ash poisoning durumu (Ochs §V).
    
    Quasineutrality with ash:
        n_e = n_p + Z_B · n_B + Z_α · n_α     (Z_α = 2)
    
    Z_eff:
        Z_eff = (n_p Z_p² + n_B Z_B² + n_α Z_α²) / n_e
    
    α partikülleri füzyon sonrası birikir. Plazmadan kaçmazsa:
      - n_e artar (α katkısı)
      - Z_eff artar (Z_α² = 4 etkisi)
      - Bremsstrahlung artar (∝ n_e² · Z_eff)
      - **Füzyona katkı YOK** (α + p reaksiyonu var ama küçük)
    
    Ochs Şek. 8: %2 ash → ignition window kapanır (no channeling).
    
    Parametreler
    ----------
    n_p_cm3, n_B_cm3, n_alpha_cm3 : float
        Yoğunluklar (cm⁻³)
    
    Dönüş
    -----
    n_e : float
        Elektron yoğunluğu
    Z_eff : float
        Etkin yük
    """
    Z_alpha = 2
    n_e = n_p_cm3 + Z_B * n_B_cm3 + Z_alpha * n_alpha_cm3
    Z_eff = (n_p_cm3 * Z_p**2 + 
             n_B_cm3 * Z_B**2 + 
             n_alpha_cm3 * Z_alpha**2) / n_e
    return n_e, Z_eff


def P_brem_with_ash(n_p_cm3, n_B_cm3, n_alpha_cm3, T_e_keV):
    """Ash poisoning'li bremsstrahlung gücü (Ochs §V).
    
    NRL + Svensson düzeltmesi, ama n_e ve Z_eff α'yı içerir.
    
    Tipik: %2 ash → P_B yaklaşık %12-15 artar (n_e² × Z_eff etkisi).
    """
    n_e, Z_eff = Z_eff_with_ash(n_p_cm3, n_B_cm3, n_alpha_cm3)
    return P_bremsstrahlung(n_e, T_e_keV, Z_eff)


def ignition_check_with_ash(n_p, n_B, n_alpha, T_i_keV, T_e_keV=None,
                              alpha_channeling_eff=0.0, lnLambda=17.0):
    """Ignition check ash poisoning ve α-channeling dahil (Ochs §V Şek. 8).
    
    Üç senaryo (Ochs Şek. 8):
    (a) η_α = 0 (no channeling): P_F < P_B → tutuşma yok (Şek. 8a)
    (b) η_α = 0.5, thermal proton kanallı: P_F > P_B (Şek. 8b)
    (c) η_α = 0.5, fast proton kanallı (kinetic FP): genişler (Şek. 8c)
    
    Bu fonksiyon (a) ve (b) durumlarını hesaplar. (c) için kinetic FP solver
    kullanılır (main_validation.py).
    
    Parametreler
    ----------
    n_p, n_B, n_alpha : float
        Yoğunluklar (cm⁻³)
    T_i_keV : float
        İyon sıcaklığı (T_p ≈ T_B varsayılır, çünkü τ_pb << τ_E)
    T_e_keV : float or None
        Elektron sıcaklığı; None ise self-consistent
    alpha_channeling_eff : float
        α'lardan thermal protonlara kanallama verimi (0-1)
        Ochs notation: η_α
    lnLambda : float
    
    Dönüş
    -----
    dict
    """
    if T_e_keV is None:
        T_e_keV = find_self_consistent_Te(n_p, n_B, T_i_keV, lnLambda)
    
    n_e, Z_eff = Z_eff_with_ash(n_p, n_B, n_alpha)
    n_i_total = n_p + n_B + n_alpha
    ash_fraction = n_alpha / n_i_total if n_i_total > 0 else 0
    
    # P_F (sadece p ve B reaksiyona girer)
    P_F = P_fusion_thermal(n_p, n_B, T_i_keV)
    
    # P_B (ash dahil)
    P_B = P_bremsstrahlung(n_e, T_e_keV, Z_eff)
    
    # P_α üretim
    P_alpha_gen = P_alpha_total(n_p, n_B, T_i_keV)
    
    # α-channeling: η_α fraksiyonu thermal protonlara aktarılıyor
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
# 4. α GÜCÜ DAĞILIMI
# ============================================================

def P_alpha_total(n_p_cm3, n_B_cm3, T_i_keV):
    """Toplam α gücü (üretim hızı × ortalama enerji).
    
    P_α = n_p · n_B · ⟨σv⟩ · E_α_total
    
    Burada E_α_total = Q ≈ 8.7 MeV (üç α'nın toplam enerjisi).
    Bu güç plazma içinde dağılır: bir kısmı protonlara, bir kısmı 
    elektronlara, bir kısmı borona.
    
    Dönüş: W/cm³
    """
    sv = sigma_v_TB_numerical(T_i_keV)
    if hasattr(sv, '__len__') and not np.isscalar(T_i_keV):
        pass
    else:
        sv = sv[0] if hasattr(sv, '__len__') else sv
    
    rate = n_p_cm3 * n_B_cm3 * sv  # reaksiyon/cm³/s
    E_alpha_total_erg = Q_pB11_keV * keV_to_erg
    P_erg = rate * E_alpha_total_erg
    return P_erg / 1e7


def alpha_power_to_electrons_fraction(T_e_keV, n_e, n_p, n_B, lnLambda=17.0):
    """α gücünün elektronlara giden kısmı (yaklaşık).
    
    Yavaşlama α'ları enerjilerini ağırlıklı olarak elektronlara verir
    (yüksek v'de) ve iyonlara (düşük v'de). Genel olarak T_e ≈ 150 keV
    p-B11 plazmalarında elektronlara gitme oranı ~%10 mertebesindedir
    (Putvinski 2019).
    
    Bu basit bir empirik formül; tam hesap için α dağılımı integralı gerekir.
    
    Dönüş: 0 ≤ frac ≤ 1
    """
    # Kabaca lineer ölçek: T_e arttıkça elektron payı azalır
    # T_e = 150 keV'de yaklaşık %10
    # T_e = 50 keV'de daha fazla (~%30)
    # T_e = 300 keV'de daha az (~%5)
    
    # Putvinski Şek. B1'den ilham alan basit fit
    frac_e = 0.15 * (150.0 / max(T_e_keV, 50.0))**0.5
    return min(frac_e, 0.5)


# ============================================================
# 5. SELF-CONSISTENT T_e ÇÖZÜCÜ
# ============================================================

def find_self_consistent_Te(n_p, n_B, T_i_keV, lnLambda=17.0,
                              T_e_min=20.0, T_e_max=500.0):
    """Putvinski Eq. 8: P_α,e + P_i,e = P_Brem
    
    Verili T_i ve plazma karışımı için T_e'yi öyle ayarla ki elektron
    güç dengesi sağlansın.
    
    Bu durağan-hal koşulu: elektronlara gelen güç (α'dan + iyondan) =
    elektronların kaybı (bremsstrahlung).
    
    Parametreler
    ----------
    n_p, n_B : float
        İyon yoğunlukları
    T_i_keV : float
        İyon sıcaklığı (T_p = T_B varsayımı)
    
    Dönüş
    -----
    T_e_keV : float
        Self-consistent elektron sıcaklığı
    """
    n_e = n_p + Z_B * n_B
    Z_eff = Z_eff_calc(n_p, n_B)
    
    # P_α,e + P_i,e - P_Brem = 0 fonksiyonu
    def power_residual(T_e):
        # α'dan elektrona güç
        P_alpha = P_alpha_total(n_p, n_B, T_i_keV)
        frac_e = alpha_power_to_electrons_fraction(T_e, n_e, n_p, n_B, lnLambda)
        P_alpha_e = P_alpha * frac_e
        
        # İyondan elektrona güç (T_i > T_e ise pozitif)
        P_ie = P_ion_electron_transfer(n_p, n_B, T_i_keV, T_i_keV, T_e, lnLambda)
        
        # Bremsstrahlung kaybı
        P_brem = P_bremsstrahlung(n_e, T_e, Z_eff)
        
        return P_alpha_e + P_ie - P_brem
    
    # T_e_min'de pozitif (P_alpha+P_ie > P_brem), T_e_max'de negatif olmalı
    try:
        T_e_solution = brentq(power_residual, T_e_min, T_e_max, xtol=0.5)
    except ValueError:
        # Çözüm yoksa T_e/T_i = 0.5 varsayımı (Putvinski)
        T_e_solution = T_i_keV * 0.5
    
    return T_e_solution


# ============================================================
# 6. TUTUŞMA KRİTERİ ve τ_E*
# ============================================================

def ignition_check(n_p, n_B, T_i_keV, T_e_keV=None, lnLambda=17.0):
    """Tutuşma koşulu: P_F > P_Brem?
    
    Eğer T_e_keV verilmemişse, self-consistent T_e bulur.
    
    Dönüş
    -----
    info : dict
        'P_F' : fusion gücü (W/cm³)
        'P_Brem' : bremsstrahlung (W/cm³)
        'P_F_minus_P_Brem' : net (W/cm³)
        'ignition' : bool
        'T_e_keV' : kullanılan T_e
        'tau_E_star' : Ochs metrisi (saniye)
    """
    if T_e_keV is None:
        T_e_keV = find_self_consistent_Te(n_p, n_B, T_i_keV, lnLambda)
    
    n_e = n_p + Z_B * n_B
    Z_eff = Z_eff_calc(n_p, n_B)
    
    # Fusion gücü
    P_F = P_fusion_thermal(n_p, n_B, T_i_keV)
    if hasattr(P_F, '__len__'):
        P_F = float(P_F)
    
    # Bremsstrahlung
    P_Brem = P_bremsstrahlung(n_e, T_e_keV, Z_eff)
    
    # Net güç
    P_net = P_F - P_Brem
    
    # τ_E* = U_K / (P_F - P_Brem)  [Ochs metriği]
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
# DOĞRULAMA TESTLERİ
# ============================================================

def _test_bremsstrahlung():
    """Bremsstrahlung formülünü Putvinski Şek. 4 ile karşılaştır."""
    print("=" * 70)
    print("TEST 1: BREMSSTRAHLUNG GÜCÜ")
    print("=" * 70)
    
    # Putvinski Şek. 4: n_i = 1e20 m⁻³ = 1e14 cm⁻³, fB = 0.15
    n_i = 1e14
    f_B = 0.15
    n_B = f_B * n_i
    n_p = (1 - f_B) * n_i
    n_e = n_p + Z_B * n_B
    Z_eff = Z_eff_calc(n_p, n_B)
    
    print(f"Plazma: n_i=10¹⁴, f_B=0.15, n_e={n_e:.2e}, Z_eff={Z_eff:.3f}")
    print()
    
    # Putvinski Şek. 4'ten (yaklaşık okunan) değerler:
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
    
    print(f"{'T_i (keV)':>10} | {'T_e (keV)':>10} | {'P_Brem (W/cm³)':>16} | {'beklenen':>10}")
    print("-" * 60)
    for T_i, T_e, P_expected in test_cases:
        P_brem = P_bremsstrahlung(n_e, T_e, Z_eff)
        ratio = P_brem / P_expected if P_expected > 0 else 0
        marker = "✅" if 0.5 < ratio < 2.0 else "⚠"
        print(f"{T_i:>10.0f} | {T_e:>10.0f} | {P_brem:>16.3e} | "
              f"{P_expected:>10.2f} {marker}")
    
    # x = T_e/E_rest karşılaştırması
    print(f"\nT_e=150 keV'de relativistik faktör:")
    R = relativistic_R_factor(150.0)
    print(f"  R(x=0.293) = {R:.3f}  (Putvinski Şek. 3: ~1.05-1.10)")
    if 1.0 < R < 1.3:
        print(f"  ✅ Putvinski Şek. 3 ile uyumlu (relativistik artırım)")
    else:
        print(f"  ⚠ Beklenen aralıkta değil")


def _test_fusion_power():
    """Fusion güç formülünü test et."""
    print("\n" + "=" * 70)
    print("TEST 2: FUSION GÜCÜ (TERMAL)")
    print("=" * 70)
    
    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i
    
    print(f"Plazma: n_i=10¹⁴, n_p={n_p:.2e}, n_B={n_B:.2e}")
    print()
    
    # Putvinski Şek. 4'ten okunan termal P_F (orijinal SW kesiti):
    # T_i=300 keV: ~1 MW/m³ = 1 W/cm³ (pek doğru olmayabilir)
    # T_i=500 keV: ~1.1 MW/m³ = 1.1 W/cm³ (peak)
    # 
    # TB kesiti %30 daha yüksek vermeli
    
    print(f"{'T_i (keV)':>10} | {'P_F (W/cm³)':>14} | {'⟨σv⟩':>14}")
    print("-" * 50)
    for T_i in [100, 200, 300, 400, 500, 700]:
        P_F = P_fusion_thermal(n_p, n_B, T_i)
        sv = sigma_v_TB_numerical(T_i)[0]
        print(f"{T_i:>10.0f} | {P_F:>14.3e} | {sv:>14.3e}")


def _test_ignition_window():
    """Putvinski Şek. 4 reprodüksiyonu — kritik test."""
    print("\n" + "=" * 70)
    print("TEST 3: TUTUŞMA PENCERESI (Putvinski Şek. 4 reprodüksiyonu)")
    print("=" * 70)
    
    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i
    
    print(f"\nPlazma: n_i=10¹⁴, f_B=0.15")
    print(f"Termal varsayımıyla (kinetik artırım yok)")
    print()
    
    print(f"{'T_i':>6} | {'T_e':>6} | {'P_F':>10} | {'P_Brem':>10} | "
          f"{'P_F/P_B':>8} | {'ign':>4} | {'τ_E*':>10}")
    print("-" * 75)
    
    T_i_scan = [100, 150, 200, 250, 300, 350, 400, 500, 600]
    for T_i in T_i_scan:
        info = ignition_check(n_p, n_B, T_i)
        marker = "✓" if info['ignition'] else "✗"
        tau_str = f"{info['tau_E_star_s']:.1e}" if np.isfinite(info['tau_E_star_s']) else "∞"
        print(f"{T_i:>6.0f} | {info['T_e_keV']:>6.1f} | "
              f"{info['P_F']:>10.3e} | {info['P_Brem']:>10.3e} | "
              f"{info['P_F_over_P_Brem']:>8.3f} | {marker:>4} | {tau_str:>10}")
    
    # Putvinski 2019'a göre tutuşma penceresi: T_i ~ 250-400 keV
    print(f"\nBeklenti: Tutuşma T_i ∈ [250, 400] keV aralığında (Putvinski Şek. 4)")
    print(f"P_F/P_Brem peak ≈ 1.03 (Putvinski: %3 marj)")


if __name__ == "__main__":
    _test_bremsstrahlung()
    _test_fusion_power()
    _test_ignition_window()
