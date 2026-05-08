"""
collision_operators.py — Trubnikov çarpışma operatörleri

Birincil referans: Putvinski et al. (2019) Ek A, Eq. A.12-A.24
Trubnikov B.A. (1965), "Particle Interactions in a Fully Ionized Plasma"

Bu modül izotropik proton test parçacığı için difüzyon (D) ve sürtünme (F)
katsayılarını sağlar. Alan parçacıkları: termal protonlar (p), boronlar (B),
elektronlar (e), ve daha sonra non-Maxwellian alfa parçacıkları (α*).

Putvinski Eq. 3:
  (1/v²)·∂_v[v²·(D_p·∂_v f_p + (F_p/m_p)·f_p)] - ν_fus·f_p + S = 0

Burada:
  D_p = D_pp + D_pB + D_pe + D*_pα   (Eq. 4)
  F_p = m_p²·v·[(D_pp + D_pB)/T + D_pe/T_e] + F*_pα   (Eq. 5)

KRİTİK DOĞRULAMA: Maxwell dağılımı verildiğinde:
  D·∂_v f_M + (F/m)·f_M = 0   (Detailed balance)
Çünkü f_M = exp(-mv²/2T) için ∂_v f_M = -(mv/T)·f_M
yani D·(-mv/T) + F/m = 0 → F = m²v·D/T  (T cinsinden)

Bu modülün her test parçacığı için detailed balance'i sağlaması gerekir.

Birim sistemi: CGS (cm, g, s, erg) — Trubnikov'un orijinal sistemi
"""

import numpy as np
from cross_sections import (
    mu_pB11_g, m_p_u, m_B11_nuc_u, amu_g,
    keV_to_erg, e_charge,
)

# ============================================================
# CGS BİRİMLERİ VE FİZİKSEL SABİTLER
# ============================================================

# Temel CGS sabitleri
m_p_g = m_p_u * amu_g           # proton kütlesi, gram
m_B_g = m_B11_nuc_u * amu_g     # B-11 çekirdek kütlesi, gram
m_e_g = 9.1093837015e-28        # elektron kütlesi, gram
e_esu = 4.80320425e-10          # elektron yükü, esu (CGS Gauss)
c_cm  = 2.99792458e10           # ışık hızı, cm/s

# Yük durumları
Z_p = 1   # proton
Z_B = 5   # B-11
Z_e = -1  # elektron (mutlak değer kullanılır kare alındığı için)


# ============================================================
# COULOMB LOGARİTMASI
# ============================================================

def coulomb_log(n_e_cm3, T_e_keV, species_pair='ee'):
    """
    Coulomb logaritması Λ. NRL Plasma Formulary 2019, sayfa 34.
    
    Parametreler
    ----------
    n_e_cm3 : float
        Elektron yoğunluğu, cm⁻³
    T_e_keV : float
        Elektron sıcaklığı, keV
    species_pair : str
        'ee' (e-e), 'ei' (e-i), 'ii' (i-i)
    
    Dönüş
    -----
    Lambda : float
        Coulomb logaritması (boyutsuz)
    
    Notlar
    -----
    p-B11 plazmaları için tipik değer Λ ≈ 15-20.
    Putvinski, makale boyunca Λ'yı tek bir değer olarak alır (parametre olarak
    verir, hesaplamaz). Biz NRL formüllerini kullanacağız ama gerektiğinde
    sabit bir değer (örn. 17) ile geçersiz kılınabilir.
    """
    T_e_eV = T_e_keV * 1000.0
    
    if species_pair == 'ee':
        # e-e: T_e ≥ 10 eV için
        Lambda = 23.5 - np.log(n_e_cm3**0.5 * T_e_eV**(-1.25)) \
                 - np.sqrt(1e-5 + (np.log(T_e_eV) - 2)**2 / 16.0)
    elif species_pair == 'ei':
        # e-i, T_e > 10 eV (relativistic olmayan)
        # Putvinski Eq. A.6: λ_ie genellikle ~15-20
        if T_e_eV < 10.0:
            Lambda = 23.0 - np.log(n_e_cm3**0.5 * Z_p * T_e_eV**(-1.5))
        else:
            Lambda = 24.0 - np.log(n_e_cm3**0.5 / T_e_eV)
    elif species_pair == 'ii':
        # i-i, basitleştirilmiş form (tek tür Z₁Z₂=1 için)
        T_i_eV = T_e_eV  # varsayım
        Lambda = 23.0 - np.log(2 * n_e_cm3**0.5 / T_i_eV**1.5)
    else:
        raise ValueError(f"Bilinmeyen species_pair: {species_pair}")
    
    # Aşırı düşük değerlere karşı koruma (zayıf çiftleşme rejimi sınırı)
    return max(Lambda, 1.0)


# ============================================================
# I_ROSENBLUTH İNTEGRALİ — Putvinski Eq. A.11
# ============================================================
# Genel form, ama Maxwellian alan parçacıkları için kapalı form var

def D_test_on_Maxwellian(v_cm_s, n_field_cm3, T_field_keV, m_field_g,
                          Z_test, Z_field, lnLambda):
    """
    Test parçacığının Maxwell alan parçacıkları üzerindeki difüzyon katsayısı.
    Putvinski Eq. A.15:
    
      D_ρσ = (4π·Λ·Z_ρ²·Z_σ²·e⁴) / (A_ρ²·A_σ·m_p³) · (4π·T_σ/v³) · ∫₀^v v'²·f_σ(v') dv'
    
    Burada f_σ Maxwell dağılımı, integral kapalı formda yazılabilir.
    
    Daha basit form (Putvinski Eq. A.22 ve civarı):
    
      D_ρσ = (4π·Λ·Z_ρ²·Z_σ²·e⁴·n_σ·T_σ) / (m_ρ²·m_σ·v) · G(x)
    
    burada G(x) bir özel fonksiyon, x = v/v_th_σ.
    
    Bu modülde GENEL formu kullanıyoruz (Putvinski Eq. A.15 + A.21):
    
      D_ρσ ≈ (4π·Λ·Z_ρ²·Z_σ²·e⁴·n_σ·T_σ) / (m_ρ²·m_p²·v³) ·
             [v³ / (v³ + (3√π/4)·v_th,σ³)]
    
    Bu interpolasyon test parçacığı tüm hızlarda geçerli (yavaş ve hızlı limit).
    
    Parametreler
    ----------
    v_cm_s : ndarray
        Test parçacığı hızları, cm/s
    n_field_cm3 : float
        Alan parçacığı yoğunluğu, cm⁻³
    T_field_keV : float
        Alan parçacığı sıcaklığı, keV
    m_field_g : float
        Alan parçacığı kütlesi, gram
    Z_test, Z_field : int
        Yük durumları
    lnLambda : float
        Coulomb logaritması
    
    Dönüş
    -----
    D : ndarray
        Difüzyon katsayısı, cm²/s³ (hız uzayında)
    """
    v = np.atleast_1d(v_cm_s).astype(float)
    
    # Termal hız (Maxwell): v_th = √(2T/m)
    T_field_erg = T_field_keV * keV_to_erg
    v_th = np.sqrt(2 * T_field_erg / m_field_g)
    
    # Test parçacığı kütlesi (proton varsayılır — bu modül izotropik test proton için)
    m_test_g = m_p_g
    
    # Önçarpan
    prefactor = (4 * np.pi * lnLambda * (Z_test * Z_field * e_esu**2)**2 *
                 n_field_cm3 * T_field_erg) / (m_test_g**2 * m_field_g)
    
    # Hız bağımlılığı: yumuşak interpolasyon (Putvinski Eq. A.21)
    # v³ / (v³ + (3√π/4)·v_th³)
    smooth_factor = v**3 / (v**3 + (3 * np.sqrt(np.pi) / 4) * v_th**3)
    
    # D = prefactor / v³ · smooth_factor = prefactor / (v³ + (3√π/4)·v_th³)
    D = prefactor / (v**3 + (3 * np.sqrt(np.pi) / 4) * v_th**3)
    
    return D


def F_test_on_Maxwellian(v_cm_s, n_field_cm3, T_field_keV, m_field_g,
                          Z_test, Z_field, lnLambda):
    """
    Test parçacığının Maxwell alan parçacıkları üzerindeki sürtünme katsayısı.
    
    Detailed balance (Putvinski Eq. 5):
      F_ρσ = m_ρ² · v · D_ρσ / T_σ
    
    Bu kesin Maxwell dengesi için zorunludur.
    Putvinski'nin Eq. A.13 ve A.14'ü buna eşdeğerdir.
    
    Dönüş
    -----
    F : ndarray
        Sürtünme katsayısı, gram·cm/s² (kuvvet)
    """
    v = np.atleast_1d(v_cm_s).astype(float)
    m_test_g = m_p_g
    
    D = D_test_on_Maxwellian(v_cm_s, n_field_cm3, T_field_keV, m_field_g,
                              Z_test, Z_field, lnLambda)
    
    T_field_erg = T_field_keV * keV_to_erg
    F = m_test_g**2 * v * D / T_field_erg
    
    return F


# ============================================================
# ELEKTRONLARLA ÇARPIŞMA — Yüksek hız limiti
# ============================================================
#
# Putvinski Eq. A.24: protonlar elektronlardan ÇOK YAVAŞ (v << v_th_e),
# bu yüzden elektronlarla çarpışma için Maxwell formülü değişir:
#
#   D_pe ≈ (8√π·Λ·e⁴·n_e) / (3·m_p²) · √(m_e/(2·T_e))
#
# Bu form v-bağımsızdır (zayıf bağımlılık). Genel formülümüz bu limiti
# yüksek v_th_e (elektronlar çok hızlı) sayesinde otomatik olarak yakalar.

def D_pp(v, n_p, T_p_keV, lnLambda=17.0):
    """Proton-proton difüzyonu (termal protonlar Maxwell varsayımı)."""
    return D_test_on_Maxwellian(v, n_p, T_p_keV, m_p_g, Z_p, Z_p, lnLambda)


def D_pB(v, n_B, T_B_keV, lnLambda=17.0):
    """Proton-Boron difüzyonu."""
    return D_test_on_Maxwellian(v, n_B, T_B_keV, m_B_g, Z_p, Z_B, lnLambda)


def D_pe(v, n_e, T_e_keV, lnLambda=17.0):
    """Proton-elektron difüzyonu (yüksek v_th_e limiti otomatik yakalanır)."""
    return D_test_on_Maxwellian(v, n_e, T_e_keV, m_e_g, Z_p, 1, lnLambda)


def F_pp(v, n_p, T_p_keV, lnLambda=17.0):
    """Proton-proton sürtünmesi."""
    return F_test_on_Maxwellian(v, n_p, T_p_keV, m_p_g, Z_p, Z_p, lnLambda)


def F_pB(v, n_B, T_B_keV, lnLambda=17.0):
    """Proton-Boron sürtünmesi."""
    return F_test_on_Maxwellian(v, n_B, T_B_keV, m_B_g, Z_p, Z_B, lnLambda)


def F_pe(v, n_e, T_e_keV, lnLambda=17.0):
    """Proton-elektron sürtünmesi.
    
    NOT: Burada T_e kullanılır (Putvinski Eq. 5'in üçüncü terimi).
    Eğer T_e < T_p ise, F_pe < F_pp (orantılı), yani elektronlara enerji
    aktarımı yavaşlar — bu hot-ion modunun kalbidir.
    """
    return F_test_on_Maxwellian(v, n_e, T_e_keV, m_e_g, Z_p, 1, lnLambda)


# ============================================================
# TOPLAM DİFÜZYON VE SÜRTÜNME (alfa katkısı hariç)
# ============================================================

def D_total_thermal(v, n_p, n_B, n_e, T_p_keV, T_B_keV, T_e_keV, lnLambda=17.0):
    """Termal alan parçacıklarından toplam difüzyon (alfa hariç)."""
    return (D_pp(v, n_p, T_p_keV, lnLambda) +
            D_pB(v, n_B, T_B_keV, lnLambda) +
            D_pe(v, n_e, T_e_keV, lnLambda))


def F_total_thermal(v, n_p, n_B, n_e, T_p_keV, T_B_keV, T_e_keV, lnLambda=17.0):
    """Termal alan parçacıklarından toplam sürtünme (alfa hariç).
    
    Putvinski Eq. 5:
      F_p = m_p²·v·[(D_pp + D_pB)/T_i + D_pe/T_e]
    
    Bu, T_i ≠ T_e olduğunda (hot-ion modu) elektron sürtünmesinin ayrı
    sıcaklıkla geldiğini sağlar.
    """
    return (F_pp(v, n_p, T_p_keV, lnLambda) +
            F_pB(v, n_B, T_B_keV, lnLambda) +
            F_pe(v, n_e, T_e_keV, lnLambda))


# ============================================================
# DOĞRULAMA: DETAILED BALANCE TESTİ
# ============================================================

def _test_detailed_balance():
    """
    Maxwell dağılımı altında D·∂_v f_M + (F/m)·f_M = 0 olmalı.
    
    Bu, çarpışma operatörünün Maxwell dengesine yol açtığını doğrular.
    Eğer bu test başarısızsa, FP solver Maxwell dağılımını korumaz ve
    kod fundamentalde bozuktur.
    """
    print("=" * 70)
    print("DETAILED BALANCE TESTİ")
    print("=" * 70)
    print("Maxwell f_M altında: D·∂_v f_M + (F/m)·f_M = 0 beklenir")
    print()
    
    # Parametreler: tipik p-B11 plazma koşulları
    n_p = 8.5e13   # cm⁻³
    n_B = 1.5e13   # cm⁻³
    n_e = n_p + Z_B * n_B  # quasineutrality
    T_p = 300.0    # keV
    T_B = 300.0    # keV
    T_e = 150.0    # keV (hot-ion modu)
    lnLambda = 17.0
    
    # Hız grid'i — termal hız civarında
    T_p_erg = T_p * keV_to_erg
    v_th_p = np.sqrt(2 * T_p_erg / m_p_g)  # cm/s
    print(f"v_th_p = {v_th_p:.3e} cm/s")
    
    v_grid = np.linspace(0.1, 5.0, 50) * v_th_p  # 0.1·v_th'den 5·v_th'ye
    
    # Maxwell dağılımı (T_p sıcaklığında)
    f_M = (m_p_g / (2 * np.pi * T_p_erg))**1.5 * np.exp(-m_p_g * v_grid**2 / (2 * T_p_erg))
    
    # Türev: ∂_v f_M = -(m·v/T)·f_M
    df_M_dv = -(m_p_g * v_grid / T_p_erg) * f_M
    
    # ÖNEMLİ: Detailed balance, T_p = T_B = T_e olduğunda kesin sağlanır.
    # Hot-ion modunda (T_e < T_p) elektron katkısı dengesizleştirir.
    
    # ÖNCELİKLE TEK SICAKLIK TESTİ (T_p = T_B = T_e = 300 keV)
    print("\n--- Test 1: Tek sıcaklık (T_p = T_B = T_e = 300 keV) ---")
    D_total = D_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_p, lnLambda)
    F_total = F_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_p, lnLambda)
    
    # FP akı: J = D·∂_v f + (F/m)·f
    # Detailed balance: J = 0
    flux = D_total * df_M_dv + (F_total / m_p_g) * f_M
    
    # Göreceli hata: |flux| / (D · |∂_v f_M|)
    rel_err = np.abs(flux) / (D_total * np.abs(df_M_dv) + 1e-300)
    
    print(f"Maks göreceli hata: {np.max(rel_err):.3e}")
    print(f"Ortalama göreceli hata: {np.mean(rel_err):.3e}")
    
    if np.max(rel_err) < 1e-10:
        print("✅ Tek sıcaklıkta detailed balance MAKİNE HASSASİYETİNDE sağlandı")
    elif np.max(rel_err) < 1e-3:
        print("✅ Tek sıcaklıkta detailed balance sağlandı (sayısal hata içinde)")
    else:
        print("❌ DETAILED BALANCE BOZUK — kod hatalı")
    
    # HOT-ION MODU TESTİ
    print("\n--- Test 2: Hot-ion modu (T_p = T_B = 300, T_e = 150 keV) ---")
    print("Bu durumda detailed balance KESİN sağlanmaz (T_e ≠ T_p).")
    print("FP akısı: J = (mv·f) · [D_pe·(1/T_e - 1/T_p)]")
    print("T_e < T_p ise (1/T_e - 1/T_p) > 0 → akı POZİTİF (yüksek-v'ye doğru)")
    print("Bu, proton kuyruğunun BÜYÜMESİNE neden olur — bu Putvinski'nin")
    print("açıkladığı %10 'kinetik fusion artırım' mekanizmasının temeli.")
    
    D_total = D_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_e, lnLambda)
    F_total = F_total_thermal(v_grid, n_p, n_B, n_e, T_p, T_p, T_e, lnLambda)
    flux = D_total * df_M_dv + (F_total / m_p_g) * f_M
    
    # Yüksek v'de akı pozitif olmalı (proton tail birikiyor)
    i_high_v = np.argmin(np.abs(v_grid - 3 * v_th_p))
    print(f"\nv = 3·v_th'de akı: {flux[i_high_v]:.3e}")
    if flux[i_high_v] > 0:
        print(f"✅ POZİTİF akı — hot-ion modu proton kuyruğunu yukarı doğru iter")
        print(f"   (Putvinski 2019 §2.1'deki kinetik artırım mekanizması)")
    else:
        print(f"❌ NEGATİF akı — bu beklenmiyor, fizik yanlış")


def _test_individual_components():
    """Her bir bileşenin (D_pp, D_pB, D_pe) ayrı ayrı doğru ölçeklendiğini test et."""
    print("\n" + "=" * 70)
    print("BİREYSEL BİLEŞEN TESTİ")
    print("=" * 70)
    
    n_test = 1e14    # cm⁻³
    T_test = 300.0   # keV
    T_p_erg = T_test * keV_to_erg
    v_th_p = np.sqrt(2 * T_p_erg / m_p_g)
    v_test = v_th_p  # termal hız civarında değerlendir
    
    print(f"\nKoşullar: n = 10¹⁴ cm⁻³, T = 300 keV, v = v_th,p = {v_th_p:.2e} cm/s\n")
    
    # Putvinski'nin tahmini değerleri (300 keV plazmada ν_ii ≈ 100 s⁻¹)
    # ν_pp ~ D_pp / v² (boyutsal analiz)
    
    D_pp_val = D_pp(v_test, n_test, T_test)[0]
    D_pB_val = D_pB(v_test, n_test * 0.176, T_test)[0]  # n_B = 0.15·n_e
    D_pe_val = D_pe(v_test, n_test * 1.75, T_test)[0]   # n_e ≈ 1.75·n_p
    
    print(f"D_pp = {D_pp_val:.3e} cm²/s³")
    print(f"D_pB = {D_pB_val:.3e} cm²/s³")
    print(f"D_pe = {D_pe_val:.3e} cm²/s³")
    
    # Beklenti: D_pB > D_pp (Z_B² = 25), D_pe << D_pp (m_e << m_p Faktör)
    ratio_BB_pp = D_pB_val / D_pp_val
    ratio_pe_pp = D_pe_val / D_pp_val
    
    print(f"\nD_pB / D_pp = {ratio_BB_pp:.2f}")
    print(f"  Beklenen: (Z_B²/A_B)·(n_B/n_p)·yumuşatma_düzeltmesi")
    print(f"           = (25/11)·0.176·(1/0.69) ≈ 0.6")
    print(f"  {'✅' if 0.3 < ratio_BB_pp < 1.5 else '⚠'}")
    
    print(f"\nD_pe / D_pp = {ratio_pe_pp:.3e}")
    print(f"  Beklenen: (n_e/n_p)·(T_e/T_p)·(m_p/m_e)·yumuşatma ~ 0.05")
    print(f"  (Düşük v'de v_th,e³ payda dominant olduğundan büyük baskı)")
    print(f"  {'✅' if 0.01 < ratio_pe_pp < 0.5 else '⚠'}")
    
    # Çarpışma frekansı tahmini
    nu_pp = D_pp_val / v_test**2
    print(f"\nν_pp ≈ D_pp/v² = {nu_pp:.3e} s⁻¹")
    print(f"  Putvinski 300 keV plazma için tipik: ν_ii ~ 10²-10³ s⁻¹")


if __name__ == "__main__":
    _test_detailed_balance()
    _test_individual_components()
