"""
cross_sections.py — p-11B füzyon kesit ve reaktivite parametrizasyonları

Birincil referans: Tentori, A. & Belloni, F. (2023)
"Revisiting p-11B fusion cross section and reactivity, and their analytic 
approximations." Nucl. Fusion 63, 086001.

Bu modül üç parametrizasyonu içerir:
  1. Tentori-Belloni (TB, 2023) — modern referans
  2. Sikora-Weller (SW, 2016) ham veri noktaları — doğrulama için
  3. Bosch-Hale (D-T) — D-T karşılaştırması için

Tüm fonksiyonlar:
  - σ(E_CM): barn cinsinden (1 barn = 1e-24 cm²)
  - ⟨σv⟩(T): cm³/s cinsinden, Maxwell-Boltzmann ortalamalı
  - E ve T: keV cinsinden

Geçerlilik aralıkları:
  σ_TB(E):       0 < E ≤ 9760 keV (CM)
  ⟨σv⟩_TB(T):    10 ≤ T ≤ 500 keV
  ⟨σv⟩_DT(T):    0.2 ≤ T ≤ 100 keV (Bosch-Hale)
"""

import numpy as np
from scipy.integrate import quad

# ============================================================
# FİZİKSEL SABİTLER
# ============================================================

# Atomik kütleler (u, AME2020)
m_H1_u  = 1.00782503207     # H-1 atomu
m_B11_u = 11.00930516       # B-11 atomu
m_He4_u = 4.00260325413     # He-4 atomu
m_e_u   = 5.48579909e-4     # elektron

# Birim dönüşümleri
u_to_MeV  = 931.49410242    # 1 u = 931.49 MeV/c²
u_to_keV  = u_to_MeV * 1000
amu_g     = 1.66053907e-24  # 1 u = 1.66e-24 g
keV_to_J  = 1.602176634e-16
keV_to_erg = 1.602176634e-9
e_charge  = 1.602176634e-19
alpha_fs  = 1.0 / 137.035999084
barn_cm2  = 1e-24

# p-B11 reaksiyon parametreleri
# Çıplak çekirdek kütleleri (elektronları çıkar)
m_p_u       = m_H1_u - m_e_u
m_B11_nuc_u = m_B11_u - 5 * m_e_u
mu_pB11_u   = (m_p_u * m_B11_nuc_u) / (m_p_u + m_B11_nuc_u)
mu_pB11_keV = mu_pB11_u * u_to_keV
mu_pB11_g   = mu_pB11_u * amu_g

# Q-değeri (atomik kütlelerle, elektronlar dengeli)
Q_pB11_keV = (m_H1_u + m_B11_u - 3 * m_He4_u) * u_to_keV  # ≈ 8681 keV

# Gamow enerjisi: E_G = (π α Z₁ Z₂)² · 2 μ c²
# TB makalesi E_G = 22.589 MeV verir
Z1, Z2 = 1, 5
E_G_keV = (np.pi * alpha_fs * Z1 * Z2)**2 * 2 * mu_pB11_keV
# Doğrulama: E_G ≈ 22589 keV olmalı


# ============================================================
# TENTORİ-BELLONİ (2023) S-FAKTÖR PARAMETRELERİ
# ============================================================
#
# Kaynak: TB Tablo 1, sağ sütun ("This work")
# Üç parçalı fit: S₁ (E ≤ 400 keV), S₂ (400 < E ≤ 668 keV), S₃ (668 < E ≤ 9760 keV)
#
# NOT: TB makalesi parametreleri MeV·b cinsinden veriyor, biz keV·b'ye çeviriyoruz
#      C_n: MeV·b/MeV^n birimleri
#      A_L: MeV·b
#      E_L, δE_L: keV
# Birim dönüşümleri makalede E keV alındığı için C₀'ı MeV·b → keV·b çevirmek gerekmez,
# çünkü TB Eq. 3'te (E/1 keV) normalize edilmiş.

# --- S₁: 0 < E ≤ 400 keV ---
# S₁(E) = C₀ + C₁·(E/1keV) + C₂·(E/1keV)² + A_L·(δE_L/2)² / [(E-E_L)² + (δE_L/2)²]
# TB Tablo 1 "This work" sütunundan:
TB_C0  = 197.0          # MeV·b (C₀, NS ile aynı)
TB_C1  = 0.269          # MeV·b/keV (NS: 0.240)
TB_C2  = 2.54e-4        # MeV·b/keV² (NS: 2.31e-4)
# 148 keV rezonansı: TB sütunu "—" (kullanılmıyor)
# SW datası 148 keV rezonansını çözünürlük yetersizliği nedeniyle kaçırdı,
# bu yüzden TB makalesi S₁'de bu rezonansı dahil etmemiştir.

# --- S₂: 400 < E ≤ 668 keV ---
# S₂(E) = D₀ + D₁·(ΔE/100keV) + D₂·(ΔE/100keV)² + D₅·(ΔE/100keV)⁵
# burada ΔE = E - 400 keV
# C₀ ≡ S₂(400 keV) süreklilik koşulu nedeniyle bağımsız değil
TB_D0  = 346.0          # keV·b (NS: 330)
TB_D1  = 150.0          # keV·b (NS: 66.1) ← BÜYÜK FARK
TB_D2  = -59.9          # keV·b (NS: -20.3)
TB_D5  = -0.460         # keV·b (NS: -1.58)

# --- S₃: 668 < E ≤ 9760 keV ---
# S₃(E) = B + Σ_{k=0..3} A_k·(δE_k/2)² / [(E-E_k)² + (δE_k/2)²]
TB_B   = 0.381          # keV·b (NS: 4.38) ← çok küçük

# Lorentzian rezonans amplitüdleri: NS'den DRAMATİK BİÇİMDE BÜYÜK
TB_A   = np.array([1.98e6, 3.89e6, 1.36e6, 3.71e6])  # keV·b
# Bunlar TB makalesinin manşet farklılığı: A₁ NS'de 5.67e5, TB'de 3.89e6 (~7×)
TB_E   = np.array([640.9, 1211.0, 2340.0, 3294.0])    # keV (rezonans pozisyonları)
TB_dE  = np.array([85.5, 414.0, 221.0, 351.0])        # keV (rezonans genişlikleri)

# Sınır değerleri
TB_E1 = 400.0    # S₁/S₂ sınırı
TB_E2 = 668.0    # S₂/S₃ sınırı
TB_E3 = 9760.0   # geçerlilik üst sınırı

# ============================================================
# BİRİM AÇIKLAMASI
# ============================================================
#
# TB Tablo 1 birimleri MeV·b. Eq. 3-5'te A_k tek başına paya yazılmış
# (klasik Lorentzian (Γ/2)² normalizasyonu YOK):
#
#   S₃(E) = B + Σ_k A_k / [(E-E_k)² + δE_k²]
#
# Peak değeri (E = E_k): S_peak = B + A_k / δE_k²
#
# Bu, "A_k" sayısının (MeV·b·keV²) birimleriyle gelmesi anlamına gelir,
# fakat (E-E_k) ve δE_k her ikisi de keV cinsinden olduğu için sonuçta
# S [MeV·b] çıkar. Tablo başlığındaki "MeV b" S-faktörünün son birimine
# gönderme yapıyor, A_k sayısının nominal birimine değil.


# ============================================================
# TB S-FAKTÖR FONKSİYONLARI
# ============================================================

def S1_TB(E_keV):
    """TB Eq. 3: Düşük enerji S-faktörü (0 < E ≤ 400 keV)
    
    TB sütununda A_L, E_L, δE_L parametreleri "—" (kullanılmıyor!).
    SW datası 148 keV rezonansını yetersiz çözünürlük nedeniyle kaçırdığı için
    TB makalesinde 148 keV rezonansı **dahil edilmemiştir** S₁'de.
    
    Çıktı: MeV·b
    """
    E = np.atleast_1d(E_keV).astype(float)
    # Sadece polinom arkaplan (C_n MeV·b cinsinden)
    # TB Tablo 1: A_L = E_L = δE_L = "—" (TB sütunu)
    S = TB_C0 + TB_C1 * E + TB_C2 * E**2
    return S


def S2_TB(E_keV):
    """TB Eq. 4: Orta enerji S-faktörü (400 < E ≤ 668 keV)"""
    E = np.atleast_1d(E_keV).astype(float)
    dE = (E - TB_E1) / 100.0  # 100 keV'lik birimlerde
    S = TB_D0 + TB_D1 * dE + TB_D2 * dE**2 + TB_D5 * dE**5
    return S


def S3_TB(E_keV):
    """TB Eq. 5: Yüksek enerji S-faktörü (668 < E ≤ 9760 keV)
    
    S₃(E) = B + Σ_{k=0..3} A_k / [(E-E_k)² + δE_k²]
    
    NOT: TB Eq. 5 formülünde Lorentzian formu klasik (δE/2)²-tabanlı değil,
    A_k doğrudan paya yazılmış. Peak değer = A_k / δE_k².
    Tüm değerler keV cinsinden enerji, MeV·b cinsinden A_k.
    
    Çıktı: MeV·b
    """
    E = np.atleast_1d(E_keV).astype(float)
    S = np.full_like(E, TB_B)
    for k in range(4):
        # TB Eq. 5: A_k / [(E-E_k)² + δE_k²] (klasik Lorentzian değil!)
        S += TB_A[k] / ((E - TB_E[k])**2 + TB_dE[k]**2)
    return S


def S_TB(E_keV):
    """Tentori-Belloni (2023) astrofiziksel S-faktör, parçalı.
    
    Parametreler
    ----------
    E_keV : float veya ndarray
        CM enerjisi (keV). Geçerlilik: 0 < E ≤ 9760 keV.
    
    Dönüş
    -----
    S : ndarray
        S-faktör (keV·barn cinsinden).
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
    """Tentori-Belloni (2023) p-11B füzyon kesiti.
    
    σ(E)[b] = S(E)[MeV·b] / E[MeV] · exp(-√(E_G/E))
    
    KRİTİK: TB Tablo 1 parametreleri MeV·b cinsinden, dolayısıyla S(E) çıktısı
    MeV·b'dir. σ hesabında E'yi MeV'a çevirmek gerekir (S/E bölmesi için).
    Exponential argümanında E_G ve E aynı birimde olduğundan herhangi birim
    kullanılabilir (oran).
    
    Parametreler
    ----------
    E_cm_keV : float veya ndarray
        CM enerjisi (keV).
    
    Dönüş
    -----
    sigma : ndarray
        Reaksiyon kesiti (barn cinsinden).
    """
    E = np.atleast_1d(E_cm_keV).astype(float)
    sigma = np.zeros_like(E)
    mask = E > 0.5
    if np.any(mask):
        S_MeVb = S_TB(E[mask])           # S, MeV·b cinsinden
        E_MeV = E[mask] / 1000.0         # E'yi MeV'a çevir
        # σ[b] = S[MeV·b] / E[MeV] · exp(-√(E_G/E))
        sigma[mask] = (S_MeVb / E_MeV) * np.exp(-np.sqrt(E_G_keV / E[mask]))
    return sigma


# ============================================================
# TB REAKTİVİTE — MAXWELL-BOLTZMANN İNTEGRASYONU
# ============================================================

def sigma_v_TB_numerical(T_keV, E_max_keV=9760, n_points=10000):
    """Tentori-Belloni σ(E) ile Maxwell-Boltzmann reaktivitesi (sayısal).
    
    ⟨σv⟩(T) = √(8/(πμ)) · (1/T)^(3/2) · ∫₀^∞ E·σ(E)·exp(-E/T) dE
    
    Parametreler
    ----------
    T_keV : float veya ndarray
        İyon sıcaklığı (keV).
    E_max_keV : float
        İntegrasyon üst sınırı.
    n_points : int
        Kuadratür nokta sayısı.
    
    Dönüş
    -----
    sigma_v : ndarray
        Reaksiyon hızı katsayısı (cm³/s).
    """
    T_arr = np.atleast_1d(T_keV).astype(float)
    result = np.zeros_like(T_arr)
    
    for i, T in enumerate(T_arr):
        if T <= 0:
            continue
        # Hibrit grid: rezonansları yakalamak için yoğun
        E_low  = np.logspace(np.log10(0.5), np.log10(50), 500)
        E_mid  = np.linspace(50, 2000, n_points // 2)
        E_high = np.linspace(2000, E_max_keV, n_points // 2)
        E_grid = np.unique(np.concatenate([E_low, E_mid, E_high]))
        
        sig_cm2 = sigma_TB(E_grid) * barn_cm2
        E_erg = E_grid * keV_to_erg
        boltz = np.exp(-E_grid / T)
        
        integrand = sig_cm2 * E_erg * boltz
        integral_val = np.trapezoid(integrand, E_erg)
        
        T_erg = T * keV_to_erg
        prefactor = np.sqrt(8.0 / (np.pi * mu_pB11_g)) / T_erg**1.5
        result[i] = prefactor * integral_val
    
    return result


# ============================================================
# BOSCH-HALE (1992) D-T REAKTİVİTESİ — KARŞILAŞTIRMA İÇİN
# ============================================================

def sigma_v_DT_BoschHale(T_keV):
    """D-T füzyonu için ⟨σv⟩, Bosch-Hale (1992) parametrizasyonu.
    
    Geçerlilik: 0.2 ≤ T ≤ 100 keV. Hata: <0.25%.
    Referans: Bosch & Hale, Nucl. Fusion 32, 611 (1992).
    
    Dönüş: cm³/s
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
# DOĞRULAMA TESTLERİ
# ============================================================

def _test_constants():
    """Temel sabitlerin doğruluğunu kontrol et."""
    print("=" * 70)
    print("FİZİKSEL SABİTLER DOĞRULAMASI")
    print("=" * 70)
    
    # Q-değeri
    Q_lit = 8681.0  # keV (TB makalesi 8.6 MeV verir, daha kesin literatür: 8.681)
    err_Q = abs(Q_pB11_keV - Q_lit) / Q_lit * 100
    status_Q = "✅" if err_Q < 0.1 else "❌"
    print(f"Q-değeri:    {Q_pB11_keV:.2f} keV  (lit: {Q_lit:.0f} keV, hata: {err_Q:.3f}%) {status_Q}")
    
    # Gamow enerjisi
    E_G_lit = 22589.0  # keV (TB makalesi)
    err_EG = abs(E_G_keV - E_G_lit) / E_G_lit * 100
    status_EG = "✅" if err_EG < 1.0 else "❌"
    print(f"E_G:         {E_G_keV:.1f} keV  (TB: {E_G_lit:.0f} keV, hata: {err_EG:.2f}%) {status_EG}")
    
    # μc²
    mu_c2_MeV = mu_pB11_keV / 1000
    mu_lit = 859.526  # MeV (TB makalesi Eq. 6 yorumu)
    err_mu = abs(mu_c2_MeV - mu_lit) / mu_lit * 100
    status_mu = "✅" if err_mu < 0.1 else "❌"
    print(f"μc²:         {mu_c2_MeV:.3f} MeV  (TB: {mu_lit:.3f} MeV, hata: {err_mu:.3f}%) {status_mu}")


def _test_cross_section():
    """TB σ(E) değerlerinin literatürle uyumunu test et."""
    print("\n" + "=" * 70)
    print("σ(E) DOĞRULAMA — TB Şekil 1(b) hedef değerleri")
    print("=" * 70)
    
    # TB Şekil 1(b)'den okunan değerler.
    # NOT: TB makalesi 148 keV rezonansını DAHİL ETMİYOR (SW datası kaçırdı).
    # Dolayısıyla 148 keV'de TB değeri NS'den ÇOK düşük olur (rezonans yok).
    targets = [
        (148,  0.005, "148 keV — TB'de rezonans yok, sadece arkaplan"),
        (300,  0.20,  "yüksekleşen yamaç"),
        (640,  1.20,  "DOMİNANT geniş rezonans (TB peak)"),
        (1000, 0.50,  "1 MeV dipli bölge"),
        (1211, 0.70,  "ikincil rezonans"),
        (2340, 0.50,  "üçüncül rezonans"),
        (5000, 0.10,  "yüksek enerji kuyruk"),
    ]
    
    print(f"{'E (keV)':>8} | {'σ_TB':>10} | {'σ_lit':>10} | {'oran':>6} | not")
    print("-" * 75)
    for E, sig_target, note in targets:
        sig = sigma_TB(E)[0]
        ratio = sig / sig_target if sig_target > 0 else 0
        marker = "✅" if 0.5 < ratio < 2.0 else "⚠"
        print(f"{E:>8.0f} | {sig:>10.4f} | {sig_target:>10.4f} | {ratio:>5.2f}× | {marker} {note}")


def _test_reactivity():
    """TB ⟨σv⟩(T) değerlerinin literatürle uyumunu test et."""
    print("\n" + "=" * 70)
    print("⟨σv⟩(T) DOĞRULAMA — TB Şekil 2 + Tablo 2 hedef değerleri")
    print("=" * 70)
    
    # TB Şek. 2(a)'dan (linear lin-skala, m³/s) okunan değerler.
    # m³/s → cm³/s dönüşümü: ×10⁶ (1 m³ = 10⁶ cm³)
    targets = [
        (50,   5.0e-18,  "düşük T sınırı"),
        (100,  5.0e-17,  ""),
        (200,  2.0e-16,  ""),
        (300,  3.5e-16,  ""),
        (400,  4.7e-16,  ""),
        (500,  5.6e-16,  "manşet (TB geçerlilik üst sınırı)"),
    ]
    
    print(f"{'T (keV)':>8} | {'⟨σv⟩_kod (cm³/s)':>18} | {'⟨σv⟩_lit':>13} | {'oran':>6} | not")
    print("-" * 75)
    for T, sv_target, note in targets:
        sv = sigma_v_TB_numerical(T)[0]
        ratio = sv / sv_target if sv_target > 0 else 0
        marker = "✅" if 0.5 < ratio < 2.0 else "⚠"
        print(f"{T:>8.0f} | {sv:>18.3e} | {sv_target:>13.2e} | {ratio:>5.2f}× | {marker} {note}")
    
    # Optimum sıcaklık
    T_scan = np.linspace(50, 1000, 100)
    sv_scan = sigma_v_TB_numerical(T_scan)
    i_max = np.argmax(sv_scan)
    print(f"\n📊 TB peak: T = {T_scan[i_max]:.0f} keV, ⟨σv⟩ = {sv_scan[i_max]:.3e} cm³/s")


def _test_DT_comparison():
    """D-T ile karşılaştırma — Bosch-Hale validation."""
    print("\n" + "=" * 70)
    print("D-T (BOSCH-HALE) DOĞRULAMA")
    print("=" * 70)
    
    # Bosch-Hale referans değerleri (Bosch & Hale 1992 Tablo VII):
    targets_DT = [
        (10,  1.13e-16),
        (20,  4.33e-16),
        (50,  8.74e-16),  # peak civarı
        (70,  8.96e-16),  # peak (Bosch-Hale)
        (100, 8.62e-16),
    ]
    
    print(f"{'T (keV)':>8} | {'⟨σv⟩_DT (cm³/s)':>18} | {'lit':>13} | {'oran':>6}")
    print("-" * 65)
    for T, sv_lit in targets_DT:
        sv = sigma_v_DT_BoschHale(T)[0]
        ratio = sv / sv_lit
        marker = "✅" if 0.95 < ratio < 1.05 else "⚠"
        print(f"{T:>8.0f} | {sv:>18.3e} | {sv_lit:>13.2e} | {ratio:>5.3f}× {marker}")


if __name__ == "__main__":
    _test_constants()
    _test_cross_section()
    _test_reactivity()
    _test_DT_comparison()
