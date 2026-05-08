"""
main_validation.py — p-11B FP-coupled tutuşma penceresi analizi

Bu modül tüm önceki modülleri birleştirir:
  - cross_sections.py (TB analitik kesit)
  - collision_operators.py (Trubnikov çarpışmalar)
  - alpha_source.py (α kaynak ve dağılım)
  - fp_solver.py (Fokker-Planck çözücü)
  - power_balance.py (güç dengesi)

Üretilen ana çıktılar:
  1. Putvinski Şek. 4 reprodüksiyonu (P_F, P_Brem vs T_i)
  2. TB kesitinin tutuşma penceresine etkisi
  3. Kinetik artırımın etkisi (FP-distorted f_p)
  4. Tutuşma haritası (T_i, f_B düzleminde)

Görsel çıktılar matplotlib ile PNG dosyalarına yazılır.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless ortam için
import matplotlib.pyplot as plt
from matplotlib import cm

# Add src/ to import path (for repo structure)
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'src'))
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from cross_sections import (
    keV_to_erg, sigma_TB, sigma_v_TB_numerical, mu_pB11_g, barn_cm2,
)
from collision_operators import (
    m_p_g, m_B_g, m_e_g, Z_p, Z_B,
    D_total_thermal, F_total_thermal,
)
from alpha_source import (
    f_alpha_slowing_down, alpha_source_rate, slowing_down_time_alpha,
    D_p_alpha_nonMaxwell, F_p_alpha_nonMaxwell,
    D_p_alpha_with_belloni, F_p_alpha_with_belloni, belloni_2021_factor,
    m_alpha_g, E_alpha_avg_keV,
)
from fp_solver import (
    make_velocity_grid, build_FP_operator, solve_steady_state,
    maxwell_distribution, fusion_burnout_rate,
)
from power_balance import (
    P_bremsstrahlung, P_fusion_thermal, P_fusion_kinetic,
    Z_eff_calc, ignition_check, find_self_consistent_Te,
    relativistic_R_factor,
)


# ============================================================
# 1. ŞEKİL: PUTVINSKI ŞEK. 4 REPRODÜKSİYONU
# ============================================================

def plot_putvinski_fig4_reproduction(save_path='putvinski_fig4_repro.png'):
    """Putvinski 2019 Şek. 4'ün TB kesitiyle reprodüksiyonu.
    
    P_F (fusion) ve P_Brem'i T_i'nin fonksiyonu olarak çiz.
    Self-consistent T_e hesabı dahil.
    Kinetik artırım (α non-Maxwellian dahil) ve Maxwell karşılaştırması.
    """
    print("\n" + "=" * 70)
    print("ŞEKİL 1: PUTVINSKI ŞEK. 4 REPRODÜKSİYONU + KİNETİK ARTIRIM")
    print("=" * 70)
    
    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i
    n_e = n_p + Z_B * n_B
    
    T_i_grid = np.linspace(100, 700, 30)
    
    P_F_thermal = np.zeros_like(T_i_grid)
    P_F_kinetic = np.zeros_like(T_i_grid)
    P_Brem_arr = np.zeros_like(T_i_grid)
    T_e_arr = np.zeros_like(T_i_grid)
    enhancement_arr = np.zeros_like(T_i_grid)
    
    print("Hesaplama: T_i taraması (termal + kinetik)...")
    for i, T_i in enumerate(T_i_grid):
        # Termal hesap
        info = ignition_check(n_p, n_B, T_i)
        P_F_thermal[i] = info['P_F']
        P_Brem_arr[i] = info['P_Brem']
        T_e_arr[i] = info['T_e_keV']
        
        # Kinetik (α dahil)
        result = fp_coupled_calculation(T_i, T_e_arr[i], n_p, n_B,
                                         include_alpha=True, n_alpha_over_ne=0.05)
        # Kinetik P_F = termal × enhancement (göreceli ölçüm)
        P_F_kinetic[i] = P_F_thermal[i] * result['kinetic_enhancement']
        enhancement_arr[i] = result['kinetic_enhancement']
    
    # Tutuşma penceresi (kinetik için)
    ratio_kin = P_F_kinetic / P_Brem_arr
    ratio_th = P_F_thermal / P_Brem_arr
    
    i_peak_kin = np.argmax(ratio_kin)
    i_peak_th = np.argmax(ratio_th)
    
    ignition_kin = ratio_kin > 1.0
    ignition_th = ratio_th > 1.0
    
    print(f"\n📊 TERMAL: peak P_F/P_Brem = {ratio_th[i_peak_th]:.3f} @ T_i = {T_i_grid[i_peak_th]:.0f} keV")
    if np.any(ignition_th):
        print(f"   Tutuşma penceresi: T_i ∈ [{T_i_grid[ignition_th].min():.0f}, {T_i_grid[ignition_th].max():.0f}] keV")
    
    print(f"\n📊 KİNETİK (α dahil): peak = {ratio_kin[i_peak_kin]:.3f} @ T_i = {T_i_grid[i_peak_kin]:.0f} keV")
    if np.any(ignition_kin):
        print(f"   Tutuşma penceresi: T_i ∈ [{T_i_grid[ignition_kin].min():.0f}, {T_i_grid[ignition_kin].max():.0f}] keV")
    
    print(f"\n📊 Karşılaştırma: Putvinski 2019 (SW, kinetik): peak ≈ 1.03 @ T_i ≈ 300 keV")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Sol: P_F termal/kinetik ve P_Brem
    ax = axes[0]
    ax.plot(T_i_grid, P_F_thermal, 'r--', lw=2, label='$P_F$ termal (Maxwell)')
    ax.plot(T_i_grid, P_F_kinetic, 'r-', lw=2.5, label='$P_F$ kinetik (FP + α)')
    ax.plot(T_i_grid, P_Brem_arr, 'b-', lw=2.5, label='$P_{Brem}$')
    ax.fill_between(T_i_grid, P_F_kinetic, P_Brem_arr,
                     where=(P_F_kinetic > P_Brem_arr),
                     alpha=0.25, color='green', label='Kinetik tutuşma penceresi')
    ax.set_xlabel('İyon sıcaklığı $T_i$ (keV)', fontsize=12)
    ax.set_ylabel('Güç yoğunluğu (W/cm³)', fontsize=12)
    ax.set_title(f'p-$^{{11}}$B Güç Dengesi: TB kesiti + α kinetik artırım\n'
                 f'$n_i=10^{{14}}$ cm$^{{-3}}$, $f_B=0.15$', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 700)
    
    # Sağ: oran karşılaştırması
    ax = axes[1]
    ax.plot(T_i_grid, ratio_th, 'orange', lw=2, linestyle='--',
            label='Termal (TB)')
    ax.plot(T_i_grid, ratio_kin, 'g-', lw=2.5,
            label='Kinetik (TB + α)')
    ax.axhline(y=1.0, color='k', linestyle=':', label='Tutuşma eşiği')
    ax.axhline(y=1.03, color='r', linestyle='--', alpha=0.5,
               label='Putvinski (SW kinetik) peak')
    ax.set_xlabel('İyon sıcaklığı $T_i$ (keV)', fontsize=12)
    ax.set_ylabel('$P_F / P_{Brem}$', fontsize=12)
    ax.set_title('Fusion-Bremsstrahlung Oranı', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 700)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Kaydedildi: {save_path}")
    
    return T_i_grid, P_F_kinetic, P_Brem_arr, T_e_arr, ratio_kin


# ============================================================
# 2. ŞEKİL: TB vs NS KESİT KARŞILAŞTIRMASI
# ============================================================

def plot_cross_section_comparison(save_path='tb_vs_ns_cross_section.png'):
    """TB analitik kesit eğrisini görselleştir."""
    print("\n" + "=" * 70)
    print("ŞEKİL 2: TB KESİT EĞRİSİ")
    print("=" * 70)
    
    E_grid = np.logspace(np.log10(50), np.log10(8000), 1000)
    sigma = sigma_TB(E_grid)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Sol: σ(E) tüm aralık
    ax = axes[0]
    ax.semilogx(E_grid, sigma, 'r-', lw=2.5, label='Tentori-Belloni 2023')
    ax.axvspan(220, 600, alpha=0.15, color='green', label='Tutuşma penceresi (T_i)')
    ax.set_xlabel('CM enerjisi $E$ (keV)', fontsize=12)
    ax.set_ylabel('Kesit $\\sigma$ (barn)', fontsize=12)
    ax.set_title('p-$^{11}$B füzyon kesiti (TB analitik fit)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(50, 8000)
    
    # Önemli rezonansları işaretle
    ax.annotate('640 keV\npeak (1.33 b)', xy=(640, 1.33), xytext=(150, 0.7),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    ax.annotate('2.34 MeV', xy=(2340, 0.67), xytext=(3500, 1.0),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    # Sağ: ⟨σv⟩(T)
    ax = axes[1]
    T_grid = np.linspace(50, 500, 30)
    sv_arr = sigma_v_TB_numerical(T_grid)
    ax.plot(T_grid, sv_arr, 'b-', lw=2.5, label='$\\langle\\sigma v\\rangle$ (TB)')
    ax.set_xlabel('İyon sıcaklığı $T_i$ (keV)', fontsize=12)
    ax.set_ylabel('$\\langle\\sigma v\\rangle$ (cm³/s)', fontsize=12)
    ax.set_title('Maxwell-ortalamalı reaktivite', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"✅ Kaydedildi: {save_path}")


# ============================================================
# 3. ŞEKİL: BORON KONSANTRASYONU TARAMASI
# ============================================================

def plot_boron_fraction_scan(save_path='boron_fraction_scan.png'):
    """Boron kesri f_B'nin tutuşma marjına etkisi."""
    print("\n" + "=" * 70)
    print("ŞEKİL 3: BORON KESRİ TARAMASI")
    print("=" * 70)
    
    n_i = 1e14
    T_i_grid = np.linspace(150, 600, 30)
    f_B_values = [0.10, 0.13, 0.15, 0.17, 0.20]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    colors = cm.viridis(np.linspace(0.15, 0.85, len(f_B_values)))
    
    # Sol: P_F/P_Brem
    ax = axes[0]
    for f_B, color in zip(f_B_values, colors):
        n_p = (1 - f_B) * n_i
        n_B = f_B * n_i
        ratio = np.zeros_like(T_i_grid)
        for i, T_i in enumerate(T_i_grid):
            info = ignition_check(n_p, n_B, T_i)
            ratio[i] = info['P_F_over_P_Brem']
        ax.plot(T_i_grid, ratio, color=color, lw=2,
                label=f'$f_B = {f_B:.2f}$')
    
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7)
    ax.set_xlabel('$T_i$ (keV)', fontsize=12)
    ax.set_ylabel('$P_F / P_{Brem}$', fontsize=12)
    ax.set_title('Boron kesrinin tutuşma marjına etkisi', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Sağ: τ_E* (Ochs metriği)
    ax = axes[1]
    for f_B, color in zip(f_B_values, colors):
        n_p = (1 - f_B) * n_i
        n_B = f_B * n_i
        tau_arr = np.zeros_like(T_i_grid)
        for i, T_i in enumerate(T_i_grid):
            info = ignition_check(n_p, n_B, T_i)
            tau_arr[i] = info['tau_E_star_s'] if np.isfinite(info['tau_E_star_s']) else np.nan
        ax.plot(T_i_grid, tau_arr, color=color, lw=2,
                label=f'$f_B = {f_B:.2f}$')
    
    ax.set_xlabel('$T_i$ (keV)', fontsize=12)
    ax.set_ylabel('$\\tau_E^*$ (s)', fontsize=12)
    ax.set_title('Gerekli enerji çekleştirme zamanı (Ochs metriği)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"✅ Kaydedildi: {save_path}")


# ============================================================
# 4. ŞEKİL: TUTUŞMA HARITASI (T_i, f_B düzleminde)
# ============================================================

def plot_ignition_map(save_path='ignition_map.png'):
    """2D tutuşma haritası: T_i × f_B düzlemi."""
    print("\n" + "=" * 70)
    print("ŞEKİL 4: 2D TUTUŞMA HARİTASI")
    print("=" * 70)
    
    n_i = 1e14
    
    T_i_grid = np.linspace(150, 600, 25)
    f_B_grid = np.linspace(0.05, 0.30, 25)
    
    ratio_map = np.zeros((len(f_B_grid), len(T_i_grid)))
    tau_map = np.zeros((len(f_B_grid), len(T_i_grid)))
    
    print("Hesaplama: 2D taraması...")
    for j, f_B in enumerate(f_B_grid):
        n_p = (1 - f_B) * n_i
        n_B = f_B * n_i
        for i, T_i in enumerate(T_i_grid):
            info = ignition_check(n_p, n_B, T_i)
            ratio_map[j, i] = info['P_F_over_P_Brem']
            tau_map[j, i] = info['tau_E_star_s'] if np.isfinite(info['tau_E_star_s']) else 1e6
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Sol: P_F/P_Brem haritası
    ax = axes[0]
    levels = [0.5, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2]
    cs = ax.contourf(T_i_grid, f_B_grid, ratio_map, levels=levels,
                      cmap='RdYlGn', extend='both')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('$P_F / P_{Brem}$', fontsize=11)
    
    # 1.0 izokontu vurgula
    ax.contour(T_i_grid, f_B_grid, ratio_map, levels=[1.0],
               colors='black', linewidths=2)
    
    ax.set_xlabel('$T_i$ (keV)', fontsize=12)
    ax.set_ylabel('Boron kesri $f_B$', fontsize=12)
    ax.set_title('Tutuşma haritası (TB kesiti)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Optimum noktayı işaretle
    j_opt, i_opt = np.unravel_index(np.argmax(ratio_map), ratio_map.shape)
    ax.plot(T_i_grid[i_opt], f_B_grid[j_opt], 'k*', markersize=18,
            markeredgecolor='white', markeredgewidth=1.5,
            label=f'Optimum: $T_i$={T_i_grid[i_opt]:.0f} keV, $f_B$={f_B_grid[j_opt]:.2f}\n'
                  f'$P_F/P_B$={ratio_map[j_opt, i_opt]:.3f}')
    ax.legend(loc='upper right', fontsize=10)
    
    # Sağ: τ_E* haritası (log skala)
    ax = axes[1]
    log_tau = np.log10(np.clip(tau_map, 1, 1e5))
    levels_tau = np.linspace(1.5, 4, 11)
    cs = ax.contourf(T_i_grid, f_B_grid, log_tau, levels=levels_tau,
                      cmap='plasma_r')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('log₁₀ $\\tau_E^*$ (s)', fontsize=11)
    
    ax.set_xlabel('$T_i$ (keV)', fontsize=12)
    ax.set_ylabel('Boron kesri $f_B$', fontsize=12)
    ax.set_title('Gerekli enerji çekleştirme zamanı', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"✅ Kaydedildi: {save_path}")
    
    print(f"\n📊 OPTİMUM NOKTA:")
    print(f"   T_i = {T_i_grid[i_opt]:.0f} keV")
    print(f"   f_B = {f_B_grid[j_opt]:.3f}")
    print(f"   P_F/P_Brem = {ratio_map[j_opt, i_opt]:.3f}")
    print(f"   τ_E* = {tau_map[j_opt, i_opt]:.1f} s")


# ============================================================
# 5. FP-COUPLED HESAP (KİNETİK ARTIRIM)
# ============================================================

def fp_coupled_calculation(T_i_keV, T_e_keV, n_p, n_B, lnLambda=17.0,
                            include_alpha=True, n_alpha_over_ne=0.02,
                            include_belloni=True,
                            verbose=False):
    """FP solver ile self-consistent f_p ve kinetik fusion gücü.
    
    Steady-state proton dağılımını çöz:
      L[f_p] - ν_fus·f_p = -S_termal
    
    α non-Maxwellian katkısı:
      D_total = D_pp + D_pB + D_pe + D*_pα · F_Belloni
      F_total = F_pp + F_pB + F_pe + F*_pα · F_Belloni
    
    Bu α katkısı, yavaşlama α'larından protonlara enerji aktarımının
    proton kuyruğuna "kaldırıcı kuvvet" eklemesini modeller.
    Putvinski'nin %10 kinetik artırım mekanizmasının kaynağı budur.
    
    Belloni 2021 elastik saçılma (include_belloni=True): α-p büyük açılı
    nükleer-Coulomb saçılma katkısı, kuyruk bölgesinde D*_pα'yı ~2× artırır.
    Bu Putvinski 2019'un %5-7 kinetik artırımını %10'a yaklaştırır.
    
    Kinetik artırım GÖRECELİ olarak ölçülür:
      enhancement = P_F[f_kinetic] / P_F[f_Maxwell]   (aynı integral ile)
    
    Parametreler
    ----------
    T_i_keV, T_e_keV : float
        İyon ve elektron sıcaklıkları
    n_p, n_B : float
        İyon yoğunlukları
    include_alpha : bool
        α katkısını dahil et (varsayılan: True)
    n_alpha_over_ne : float
        Steady-state α yoğunluk oranı n_α/n_e. FİZİKSEL parametre:
        - 0.01 (Ochs senaryosu): aktif kül kaldırma, τ_s ≈ 1 s
        - 0.02 (default): orta seviye, makul kül birikimi
        - 0.10 (Putvinski Şek. B1): Putvinski'nin senaryosu
        - 0.15-0.20: ash buildup, zehirleme tehlikesi (Ochs §V)
    include_belloni : bool
        Belloni 2021 elastik saçılma katkısını ekle (varsayılan: True).
        Kuyrukta D*_pα'yı ~2× artırır.
    
    Dönüş
    -----
    result : dict
        v_grid, f_p, f_M, P_F_thermal, P_F_max_lab, P_F_kinetic,
        kinetic_enhancement, alpha_contribution_ratio
    """
    n_e = n_p + Z_B * n_B
    
    # Hız grid (proton için)
    T_i_erg = T_i_keV * keV_to_erg
    v_th_p = np.sqrt(2 * T_i_erg / m_p_g)
    v_grid, dv = make_velocity_grid(0.05*v_th_p, 5*v_th_p, N=300)
    
    # Termal çarpışma katsayıları (T_p = T_B = T_i, T_e farklı)
    D_thermal = D_total_thermal(v_grid, n_p, n_B, n_e, T_i_keV, T_i_keV, T_e_keV, lnLambda)
    F_thermal = F_total_thermal(v_grid, n_p, n_B, n_e, T_i_keV, T_i_keV, T_e_keV, lnLambda)
    
    # α NON-MAXWELLIAN KATKISI
    D_alpha = np.zeros_like(v_grid)
    F_alpha = np.zeros_like(v_grid)
    
    if include_alpha:
        # 1. α üretim hızı
        S_alpha_total = alpha_source_rate(n_p, n_B, T_i_keV)
        
        # 2. α yavaşlama dağılımı (Putvinski Eq. B.10 şekli)
        E_birth_erg = E_alpha_avg_keV * keV_to_erg
        v_birth = np.sqrt(2 * E_birth_erg / m_alpha_g)
        
        # α grid (proton grid'den farklı, daha yüksek)
        v_alpha_grid = np.linspace(0.05*v_birth, 1.2*v_birth, 200)
        
        f_alpha_dist = f_alpha_slowing_down(
            v_alpha_grid, S_alpha_total, n_e, T_e_keV,
            n_p, n_B, T_i_keV, T_i_keV,
            v_birth_cm_s=v_birth, lnLambda=lnLambda
        )
        
        # FİZİKSEL NORMALIZE: α dağılımının ŞEKLİ Putvinski Eq. B.10'dan,
        # ama YOĞUNLUĞU n_α/n_e parametresiyle ayarla.
        # Bu, τ_s belirsizliğini ve kül kaldırma fiziği yokluğunu birleştirir.
        n_alpha_target = n_alpha_over_ne * n_e
        n_alpha_calculated = np.trapezoid(4*np.pi*v_alpha_grid**2*f_alpha_dist, v_alpha_grid)
        
        if n_alpha_calculated > 0:
            scale = n_alpha_target / n_alpha_calculated
            f_alpha_dist = f_alpha_dist * scale
        
        # 3. α'dan proton'a difüzyon ve sürtünme
        # Tam Trubnikov-Rosenbluth-MacDonald-Judd formu opsiyonel:
        # USE_FULL_TRUBNIKOV=1 environment variable ile aktif edilir
        # Maxwell-eşdeğer (varsayılan): hızlı, T_α_eff yaklaşımı
        # Tam Trubnikov: %2-9 daha doğru kuyrukta (Helander-Sigmar Eq. 3.42)
        import os
        _use_full_trub = os.environ.get('USE_FULL_TRUBNIKOV', '0') == '1'
        
        if include_belloni:
            D_alpha = D_p_alpha_with_belloni(v_grid, v_alpha_grid, f_alpha_dist, 
                                              T_i_keV, lnLambda, 
                                              use_full_trubnikov=_use_full_trub)
            F_alpha = F_p_alpha_with_belloni(v_grid, v_alpha_grid, f_alpha_dist,
                                              T_i_keV, lnLambda,
                                              use_full_trubnikov=_use_full_trub)
        else:
            D_alpha = D_p_alpha_nonMaxwell(v_grid, v_alpha_grid, f_alpha_dist, lnLambda)
            F_alpha = F_p_alpha_nonMaxwell(v_grid, v_alpha_grid, f_alpha_dist, lnLambda)
    
    # Toplam D ve F
    D_total = D_thermal + D_alpha
    F_total = F_thermal + F_alpha
    
    # Fonksiyonel formlar (Maxwell korunumu için yarı-noktada doğrudan hesap)
    # NOT: α katkısı non-Maxwell olduğu için, termal kısım için fonksiyonel form
    # geçirebiliriz; α kısmı zaten lineer interpolasyonla yarı-noktaya gelir.
    # Ama sadece termal D ve F functional için, α katkısını grid değeriyle ekleriz.
    # En basit: pür termal (alpha yokken) tam fonksiyonel; α varsa hibrit.
    if include_alpha:
        # Hibrit: D_thermal functional + D_alpha grid (yarı-noktada lineer)
        # α kontribüsyonu non-Maxwell olduğundan exact preservation zaten beklenmez,
        # ama termal Maxwell korunması yine de iyileşir.
        D_thermal_grid = D_thermal.copy()
        F_thermal_grid = F_thermal.copy()
        # Yarı-noktada α katkısını da lineer interpole et
        D_alpha_at_half = lambda v: np.interp(v, v_grid, D_alpha)
        F_alpha_at_half = lambda v: np.interp(v, v_grid, F_alpha)
        D_func_total = lambda v: D_total_thermal(v, n_p, n_B, n_e, T_i_keV, T_i_keV, T_e_keV, lnLambda) + D_alpha_at_half(v)
        F_func_total = lambda v: F_total_thermal(v, n_p, n_B, n_e, T_i_keV, T_i_keV, T_e_keV, lnLambda) + F_alpha_at_half(v)
    else:
        # Pür termal: tam fonksiyonel (Maxwell ~10⁻¹¹ korunum)
        D_func_total = lambda v: D_total_thermal(v, n_p, n_B, n_e, T_i_keV, T_i_keV, T_e_keV, lnLambda)
        F_func_total = lambda v: F_total_thermal(v, n_p, n_B, n_e, T_i_keV, T_i_keV, T_e_keV, lnLambda)
    
    # Burnout terimi
    nu_fus = fusion_burnout_rate(v_grid, n_B)
    
    # Kaynak: Maxwell şekilli (proton ısıtması — çekirdek dengesini sürdür)
    f_M_at_T_i = maxwell_distribution(v_grid, n_p, T_i_keV, m_p_g)
    nu_typ = D_thermal[len(v_grid)//2] / v_grid[len(v_grid)//2]**2
    S = nu_typ * f_M_at_T_i
    
    # FP çözümü (α dahil) — fonksiyonel D, F ile yarı-noktada doğrudan hesap
    f_p = solve_steady_state(v_grid, D_total, F_total, S, m_p_g, nu_fus=nu_fus,
                              D_func=D_func_total, F_func=F_func_total)
    
    # Yoğunluğu n_p'ye normalize et
    norm_kin = np.trapezoid(4 * np.pi * v_grid**2 * f_p, v_grid)
    if norm_kin > 0:
        f_p = f_p * (n_p / norm_kin)
    
    # KİNETİK P_F: lab-frame integral, FP-distorted f_p ile
    P_F_kin = P_fusion_kinetic(v_grid, f_p, n_B, T_i_keV)
    
    # MAXWELL REFERANSI: AYNI lab-frame integral ile (sistematik %6 sapmayı iptal eder)
    P_F_max_lab = P_fusion_kinetic(v_grid, f_M_at_T_i, n_B, T_i_keV)
    
    # Tam Maxwell-Boltzmann ⟨σv⟩ (bilgi amaçlı)
    P_F_th_MB = P_fusion_thermal(n_p, n_B, T_i_keV)
    if hasattr(P_F_th_MB, '__len__'):
        P_F_th_MB = float(P_F_th_MB)
    
    # GÖRECELİ kinetik artırım
    enhancement = P_F_kin / P_F_max_lab if P_F_max_lab > 0 else 1.0
    
    # α'nın D'ye katkı oranı (tanı amaçlı)
    if include_alpha:
        i_tail = int(0.7 * len(v_grid))  # 3·v_th civarı, kuyruk
        alpha_ratio = D_alpha[i_tail] / D_thermal[i_tail] if D_thermal[i_tail] > 0 else 0
    else:
        alpha_ratio = 0.0
    
    if verbose:
        print(f"  T_i={T_i_keV:.0f}, T_e={T_e_keV:.0f}: "
              f"P_F_kin={P_F_kin:.3e}, P_F_M={P_F_max_lab:.3e}, "
              f"enh={enhancement:.4f}, α_ratio={alpha_ratio:.3f}")
    
    return {
        'v_grid': v_grid,
        'f_p': f_p,
        'f_M': f_M_at_T_i,
        'P_F_thermal': P_F_th_MB,
        'P_F_max_lab': P_F_max_lab,
        'P_F_kinetic': P_F_kin,
        'kinetic_enhancement': enhancement,
        'alpha_contribution_ratio': alpha_ratio,
        'D_thermal': D_thermal,
        'D_alpha': D_alpha,
        'F_thermal': F_thermal,
        'F_alpha': F_alpha,
    }


def plot_kinetic_distortion(save_path='kinetic_distortion.png'):
    """FP-distorted proton dağılımı vs Maxwell.
    
    Hot-ion modunda (T_e < T_i) proton kuyruğunun kinetik artırımını göster.
    """
    print("\n" + "=" * 70)
    print("ŞEKİL 5: KİNETİK PROTON KUYRUĞU (FP vs Maxwell)")
    print("=" * 70)
    
    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i
    
    T_i = 300.0
    T_e_values = [300.0, 200.0, 150.0]  # tek sıcaklık → hot-ion
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    colors = ['blue', 'orange', 'red']
    
    ax = axes[0]
    for T_e, color in zip(T_e_values, colors):
        result = fp_coupled_calculation(T_i, T_e, n_p, n_B, 
                                         n_alpha_over_ne=0.05, include_belloni=True,
                                         verbose=True)
        v_grid = result['v_grid']
        f_p = result['f_p']
        f_M = result['f_M']
        
        T_i_erg = T_i * keV_to_erg
        v_th = np.sqrt(2 * T_i_erg / m_p_g)
        
        ax.semilogy(v_grid/v_th, f_p, color=color, lw=2,
                    label=f'$T_i={T_i:.0f}$, $T_e={T_e:.0f}$ keV')
    
    # Referans Maxwell
    f_M_ref = maxwell_distribution(v_grid, n_p, T_i, m_p_g)
    ax.semilogy(v_grid/v_th, f_M_ref, 'k--', lw=1.5,
                label=f'Maxwell ($T={T_i:.0f}$ keV)')
    
    ax.set_xlabel('$v / v_{th,p}$', fontsize=12)
    ax.set_ylabel('$f_p(v)$ (cm$^{-6}$ s$^3$)', fontsize=12)
    ax.set_title('FP proton dağılımı (hot-ion etkisi)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 4)
    
    # Sağ: kinetik artırım
    ax = axes[1]
    T_i_grid = np.linspace(150, 500, 12)
    enh_putv = np.zeros_like(T_i_grid)
    enh_bell = np.zeros_like(T_i_grid)
    
    for i, T_i in enumerate(T_i_grid):
        # Self-consistent T_e
        T_e_sc = find_self_consistent_Te(n_p, n_B, T_i)
        # Putvinski-only (Trubnikov)
        r_p = fp_coupled_calculation(T_i, T_e_sc, n_p, n_B,
                                       n_alpha_over_ne=0.05, include_belloni=False)
        enh_putv[i] = r_p['kinetic_enhancement']
        # Belloni dahil
        r_b = fp_coupled_calculation(T_i, T_e_sc, n_p, n_B,
                                       n_alpha_over_ne=0.05, include_belloni=True)
        enh_bell[i] = r_b['kinetic_enhancement']
    
    ax.plot(T_i_grid, enh_putv, 'b--', lw=2, marker='s', markersize=6,
            label='Putvinski (Trubnikov α)')
    ax.plot(T_i_grid, enh_bell, 'g-', lw=2.5, marker='o', markersize=8,
            label='+ Belloni 2021 elastik')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7)
    ax.axhline(y=1.10, color='r', linestyle='--', alpha=0.5,
               label='Putvinski %10 (hedef)')
    ax.set_xlabel('$T_i$ (keV)', fontsize=12)
    ax.set_ylabel('Kinetik artırım: $P_F^{kin} / P_F^{Max}$', fontsize=12)
    ax.set_title('Belloni 2021 elastik saçılma katkısı', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"✅ Kaydedildi: {save_path}")
    
    print(f"\nKinetik artırım özeti (self-consistent T_e):")
    print(f"{'T_i (keV)':>10} | {'Putvinski only':>15} | {'+ Belloni':>12}")
    print("-" * 45)
    for T_i, ep, eb in zip(T_i_grid, enh_putv, enh_bell):
        print(f"{T_i:>10.0f} | {ep:>15.4f} | {eb:>12.4f}")


def plot_sensitivity_analysis(save_path='sensitivity_analysis.png'):
    """Hassasiyet analizi: belirsiz parametrelerin sonuçlara etkisi.
    
    Üç ana belirsizlik kaynağı:
    1. n_α/n_e (α yoğunluk oranı) — τ_s belirsizliğini gizler
    2. Boron sıcaklığı T_B (T_p ile eşit aldık, ama gerçekte ayrı olabilir)
    3. Coulomb logaritması Λ (genelde 15-20 aralığı)
    
    Her parametre için ana metrikleri (peak P_F/P_Brem, tutuşma penceresi alt
    sınırı, optimum T_i) bir aralık üzerinde tarayacağız.
    """
    print("\n" + "=" * 70)
    print("ŞEKİL 6: HASSASİYET ANALİZİ")
    print("=" * 70)
    
    n_i = 1e14
    f_B = 0.15
    n_p = (1 - f_B) * n_i
    n_B = f_B * n_i
    n_e = n_p + Z_B * n_B
    
    # Sabit T_i grid
    T_i_grid = np.linspace(100, 700, 25)
    
    # ---- (1) n_α/n_e taraması ----
    print("\n(1/3) α yoğunluk oranı taraması...")
    n_alpha_values = [0.01, 0.02, 0.05, 0.10, 0.15]
    ratio_n_alpha = {}
    
    for n_alpha in n_alpha_values:
        ratios = []
        for T_i in T_i_grid:
            T_e_sc = find_self_consistent_Te(n_p, n_B, T_i)
            P_brem = P_bremsstrahlung(n_e, T_e_sc, Z_eff_calc(n_p, n_B))
            r = fp_coupled_calculation(T_i, T_e_sc, n_p, n_B,
                                         n_alpha_over_ne=n_alpha,
                                         include_belloni=True)
            P_F_th = P_fusion_thermal(n_p, n_B, T_i)
            if hasattr(P_F_th, '__len__'):
                P_F_th = float(P_F_th)
            P_F_kin = P_F_th * r['kinetic_enhancement']
            ratios.append(P_F_kin / P_brem)
        ratio_n_alpha[n_alpha] = np.array(ratios)
    
    # ---- (2) Belloni faktörü maksimum değeri ----
    # Belloni'nin tanh fit'i max=2.0'a doygun olur. Eğer 1.5 (zayıf) veya 
    # 2.5 (güçlü) olsa ne olur?
    print("(2/3) Belloni doubling faktörü taraması...")
    # Bu daha karmaşık, kodda değişiklik gerek. Basitleştirme: include_belloni var/yok
    
    ratios_no_belloni = []
    ratios_with_belloni = []
    for T_i in T_i_grid:
        T_e_sc = find_self_consistent_Te(n_p, n_B, T_i)
        P_brem = P_bremsstrahlung(n_e, T_e_sc, Z_eff_calc(n_p, n_B))
        P_F_th = P_fusion_thermal(n_p, n_B, T_i)
        if hasattr(P_F_th, '__len__'):
            P_F_th = float(P_F_th)
        
        r_nb = fp_coupled_calculation(T_i, T_e_sc, n_p, n_B,
                                        n_alpha_over_ne=0.05, include_belloni=False)
        r_b = fp_coupled_calculation(T_i, T_e_sc, n_p, n_B,
                                       n_alpha_over_ne=0.05, include_belloni=True)
        ratios_no_belloni.append(P_F_th * r_nb['kinetic_enhancement'] / P_brem)
        ratios_with_belloni.append(P_F_th * r_b['kinetic_enhancement'] / P_brem)
    
    ratios_no_belloni = np.array(ratios_no_belloni)
    ratios_with_belloni = np.array(ratios_with_belloni)
    
    # Termal referans
    ratios_thermal = []
    for T_i in T_i_grid:
        T_e_sc = find_self_consistent_Te(n_p, n_B, T_i)
        P_brem = P_bremsstrahlung(n_e, T_e_sc, Z_eff_calc(n_p, n_B))
        P_F_th = P_fusion_thermal(n_p, n_B, T_i)
        if hasattr(P_F_th, '__len__'):
            P_F_th = float(P_F_th)
        ratios_thermal.append(P_F_th / P_brem)
    ratios_thermal = np.array(ratios_thermal)
    
    # ---- (3) Coulomb logaritması Λ taraması ----
    print("(3/3) Coulomb logaritması taraması...")
    Lambda_values = [13, 15, 17, 19, 21]
    ratio_Lambda = {}
    
    for Lam in Lambda_values:
        ratios = []
        for T_i in T_i_grid:
            T_e_sc = find_self_consistent_Te(n_p, n_B, T_i, lnLambda=Lam)
            P_brem = P_bremsstrahlung(n_e, T_e_sc, Z_eff_calc(n_p, n_B))
            r = fp_coupled_calculation(T_i, T_e_sc, n_p, n_B,
                                         lnLambda=Lam, n_alpha_over_ne=0.05,
                                         include_belloni=True)
            P_F_th = P_fusion_thermal(n_p, n_B, T_i)
            if hasattr(P_F_th, '__len__'):
                P_F_th = float(P_F_th)
            P_F_kin = P_F_th * r['kinetic_enhancement']
            ratios.append(P_F_kin / P_brem)
        ratio_Lambda[Lam] = np.array(ratios)
    
    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
    # Panel 1: n_α/n_e duyarlılığı
    ax = axes[0]
    colors_a = cm.plasma(np.linspace(0.15, 0.85, len(n_alpha_values)))
    for n_a, color in zip(n_alpha_values, colors_a):
        label = f'$n_\\alpha/n_e = {n_a:.2f}$'
        if n_a == 0.01:
            label += ' (Ochs kül kaldırma)'
        elif n_a == 0.10:
            label += ' (Putvinski Şek. B1)'
        ax.plot(T_i_grid, ratio_n_alpha[n_a], color=color, lw=2, label=label)
    
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7)
    ax.set_xlabel('$T_i$ (keV)', fontsize=11)
    ax.set_ylabel('$P_F / P_{Brem}$', fontsize=11)
    ax.set_title('(a) α yoğunluk hassasiyeti', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.5)
    
    # Panel 2: Belloni etkisi (varsayılı/yok)
    ax = axes[1]
    ax.plot(T_i_grid, ratios_thermal, 'orange', linestyle='--', lw=2,
            label='Termal (FP yok)')
    ax.plot(T_i_grid, ratios_no_belloni, 'b-', lw=2,
            label='Putvinski only (Trubnikov)')
    ax.plot(T_i_grid, ratios_with_belloni, 'g-', lw=2.5,
            label='+ Belloni 2021 elastik')
    ax.fill_between(T_i_grid, ratios_no_belloni, ratios_with_belloni,
                     alpha=0.2, color='green', label='Belloni katkısı')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7)
    ax.set_xlabel('$T_i$ (keV)', fontsize=11)
    ax.set_ylabel('$P_F / P_{Brem}$', fontsize=11)
    ax.set_title('(b) Kinetik fizik düzeyleri', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Coulomb logaritması
    ax = axes[2]
    colors_l = cm.viridis(np.linspace(0.15, 0.85, len(Lambda_values)))
    for Lam, color in zip(Lambda_values, colors_l):
        ax.plot(T_i_grid, ratio_Lambda[Lam], color=color, lw=2,
                label=f'$\\ln\\Lambda = {Lam}$')
    
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7)
    ax.set_xlabel('$T_i$ (keV)', fontsize=11)
    ax.set_ylabel('$P_F / P_{Brem}$', fontsize=11)
    ax.set_title('(c) Coulomb log hassasiyeti', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Kaydedildi: {save_path}")
    
    # Özet tablosu
    print("\n📊 HASSASİYET ÖZETİ (peak P_F/P_Brem):")
    print(f"{'Parametre':<30} | {'Min':>6} | {'Max':>6} | {'Aralık':>8}")
    print("-" * 60)
    
    peaks_n_alpha = [np.max(ratio_n_alpha[n_a]) for n_a in n_alpha_values]
    print(f"{'n_α/n_e ∈ [0.01, 0.15]':<30} | "
          f"{min(peaks_n_alpha):>6.3f} | {max(peaks_n_alpha):>6.3f} | "
          f"±{(max(peaks_n_alpha)-min(peaks_n_alpha))/2:>6.3f}")
    
    print(f"{'Belloni var/yok':<30} | "
          f"{np.max(ratios_no_belloni):>6.3f} | {np.max(ratios_with_belloni):>6.3f} | "
          f"±{(np.max(ratios_with_belloni)-np.max(ratios_no_belloni))/2:>6.3f}")
    
    peaks_Lambda = [np.max(ratio_Lambda[L]) for L in Lambda_values]
    print(f"{'ln Λ ∈ [13, 21]':<30} | "
          f"{min(peaks_Lambda):>6.3f} | {max(peaks_Lambda):>6.3f} | "
          f"±{(max(peaks_Lambda)-min(peaks_Lambda))/2:>6.3f}")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs('/home/claude/figures', exist_ok=True)
    
    print("\n" + "█" * 70)
    print("p-11B FP-COUPLED TUTUŞMA ANALİZİ")
    print("█" * 70)
    print("\nTüm modüllerin entegre testi başlıyor...\n")
    
    # Şekil 1: Putvinski Şek. 4 reprodüksiyonu
    T_i, P_F, P_Brem, T_e, ratio = plot_putvinski_fig4_reproduction(
        '/home/claude/figures/01_putvinski_fig4_repro.png'
    )
    
    # Şekil 2: TB kesit eğrisi
    plot_cross_section_comparison(
        '/home/claude/figures/02_tb_cross_section.png'
    )
    
    # Şekil 3: Boron kesri taraması
    plot_boron_fraction_scan(
        '/home/claude/figures/03_boron_fraction_scan.png'
    )
    
    # Şekil 4: 2D tutuşma haritası
    plot_ignition_map(
        '/home/claude/figures/04_ignition_map.png'
    )
    
    # Şekil 5: Kinetik artırım
    plot_kinetic_distortion(
        '/home/claude/figures/05_kinetic_distortion.png'
    )
    
    # Şekil 6: Hassasiyet analizi (YENİ)
    plot_sensitivity_analysis(
        '/home/claude/figures/06_sensitivity_analysis.png'
    )
    
    print("\n" + "█" * 70)
    print("ANALİZ TAMAMLANDI — 6 figür üretildi")
    print("█" * 70)
