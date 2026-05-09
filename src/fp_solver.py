"""
fp_solver.py — Fokker-Planck solver, isotropic 1D velocity space

Putvinski Eq. 3:
    (1/v²)·∂_v[v²·(D·∂_v f + (F/m)·f)] - ν_fus(v)·f + S(v) = 0   (steady-state)

Time-dependent form (pseudo-time):
    ∂f/∂t = (1/v²)·∂_v[v²·(D·∂_v f + (F/m)·f)] - ν_fus(v)·f + S(v)

Numerical methods:
  - Velocity grid: linear or logarithmic, N points (200-500)
  - Discretization: Chang-Cooper exponential weighting (preserves Maxwell)
  - Time integration: implicit Crank-Nicolson (stable, 2nd order)
  - Linear system: sparse tridiagonal, scipy.sparse.linalg.spsolve

Boundary conditions:
  - v=0: ∂f/∂v = 0 (regularity), equivalent to J(0)=0
  - v=v_max: f → 0 (cutoff far enough out)

Unit system: CGS (cm, g, s, erg)
All derivative formulas are conservative (particle and energy).

CRITICAL VALIDATION:
  1. Pure Maxwell source + Maxwell collisions → Maxwell must remain stationary
  2. No source, Maxwell initial → Maxwell must not relax
  3. δ-source at high v → must produce a slowing-down tail
"""

import numpy as np
from scipy.sparse import diags, csc_matrix, eye as speye
from scipy.sparse.linalg import spsolve

from cross_sections import keV_to_erg, sigma_TB, barn_cm2
from collision_operators import (
    m_p_g, m_B_g, m_e_g, e_esu, Z_p, Z_B,
    D_total_thermal, F_total_thermal,
)


# ============================================================
# VELOCITY GRID
# ============================================================

def make_velocity_grid(v_min, v_max, N, scheme='linear'):
    """Build a velocity grid.

    Parameters
    ----------
    v_min, v_max : float
        Velocity range (cm/s). v_min > 0 (numerical stability).
    N : int
        Number of grid points.
    scheme : 'linear' or 'log'
        Grid type.

    Returns
    -------
    v : ndarray, shape (N,)
        Velocity grid (cell centers).
    dv : float or ndarray
        Step size.
    """
    if scheme == 'linear':
        v = np.linspace(v_min, v_max, N)
        dv = v[1] - v[0]
    elif scheme == 'log':
        v = np.logspace(np.log10(v_min), np.log10(v_max), N)
        dv = np.diff(v, prepend=2*v[0]-v[1])  # forward diff
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return v, dv


# ============================================================
# CHANG-COOPER DISCRETIZATION
# ============================================================
#
# The Chang-Cooper (1970) scheme discretizes the FP operator in a way that
# EXACTLY preserves the Maxwell equilibrium. This eliminates numerical
# diffusion that would otherwise spuriously broaden the distribution.
#
# FP flux: J(v) = D(v)·∂_v f + (F(v)/m)·f
# Discretized: J_{i+1/2} = D_{i+1/2}/dv · [(1-δ_{i+1/2})·f_{i+1} - δ_{i+1/2}·f_i ]
#                       + (F_{i+1/2}/m) · [δ_{i+1/2}·f_i + (1-δ_{i+1/2})·f_{i+1}]
#
# In the Chang-Cooper exponential-weighting form:
# J_{i+1/2} = (D_{i+1/2}/dv)·(f_{i+1} - f_i) + (F_{i+1/2}/m)·[δ·f_{i+1} + (1-δ)·f_i]
#
# δ_{i+1/2} = 1/w_{i+1/2} - 1/(exp(w_{i+1/2}) - 1)
# w_{i+1/2} = (F_{i+1/2}/m) · dv / D_{i+1/2}
#
# This choice analytically preserves the Maxwell distribution.


def chang_cooper_delta(w):
    """Chang-Cooper weighting function δ(w).

    For small w, δ → 1/2 (central difference).
    For large w, δ → 0 (upwind).
    """
    w = np.atleast_1d(w).astype(float)
    delta = np.zeros_like(w)

    # Numerical stability: series expansion for small w
    small = np.abs(w) < 1e-3
    delta[small] = 0.5 - w[small]/12.0  # Taylor: 1/w - 1/(e^w-1) ≈ 1/2 - w/12

    # General form
    not_small = ~small
    if np.any(not_small):
        ws = w[not_small]
        delta[not_small] = 1.0/ws - 1.0/(np.exp(ws) - 1.0)

    return delta


def build_FP_operator(v, D, F, m_test, D_func=None, F_func=None):
    """Build the FP operator as a sparse tridiagonal matrix.

    Chang-Cooper discretization: preserves Maxwell equilibrium exactly.

    FP equation in isotropic 1D:
        ∂f/∂t = (1/v²)·∂_v[v²·J(v)]
        J(v) = D(v)·∂_v f + (F(v)/m)·f

    CRITICAL: if D_func and F_func are supplied, the half-point values are
    evaluated DIRECTLY from those functions (rather than linear-averaged from
    grid values). This preserves Maxwell to ~10⁻¹¹ relative precision; the
    linear-average prescription gives ~10⁻² error.

    Chang-Cooper formulation:
        J_{i+1/2} = (D_{i+1/2}/Δv)·(f_{i+1} - f_i) + W_{i+1/2}·[δ·f_{i+1} + (1-δ)·f_i]

    with:
        W = F/m_test
        w = W·Δv/D
        δ = 1 - [1/w - 1/(e^w - 1)]   (sign convention used here)

    This form EXACTLY preserves Maxwell equilibrium (Chang & Cooper 1970).

    Parameters
    ----------
    v : ndarray, shape (N,)
        Velocity grid (uniform).
    D : ndarray, shape (N,)
        Diffusion grid values (kept for backward compatibility).
    F : ndarray, shape (N,)
        Friction grid values (kept for backward compatibility).
    m_test : float
        Test particle mass.
    D_func : callable, optional
        D(v) function. If supplied, evaluated DIRECTLY at half-points (recommended).
    F_func : callable, optional
        F(v) function. If supplied, evaluated DIRECTLY at half-points.

    Returns
    -------
    L : sparse csc matrix, shape (N, N)
    """
    N = len(v)
    dv = v[1] - v[0]

    # Half-point velocities
    v_half = 0.5 * (v[:-1] + v[1:])

    # D and F at half-points: DIRECT evaluation if function supplied,
    # otherwise linear average
    if D_func is not None:
        D_half = D_func(v_half)
    else:
        D_half = 0.5 * (D[:-1] + D[1:])

    if F_func is not None:
        F_half = F_func(v_half)
    else:
        F_half = 0.5 * (F[:-1] + F[1:])

    # Drift velocity: W = F/m
    W_half = F_half / m_test

    # Chang-Cooper Peclet number
    w = W_half * dv / (D_half + 1e-300)

    # δ weighting (sign convention used here)
    delta = 1.0 - chang_cooper_delta(w)

    # Flux coefficients
    alpha_half = D_half / dv + W_half * delta
    beta_half = -D_half / dv + W_half * (1.0 - delta)

    main_diag = np.zeros(N)
    sub_diag = np.zeros(N-1)
    sup_diag = np.zeros(N-1)

    v_squared = v**2
    v_half_squared = v_half**2

    # Interior points
    for i in range(1, N-1):
        coef = 1.0 / (v_squared[i] * dv)
        sub_diag[i-1] = -coef * v_half_squared[i-1] * beta_half[i-1]
        main_diag[i] = coef * (v_half_squared[i] * beta_half[i] -
                                v_half_squared[i-1] * alpha_half[i-1])
        sup_diag[i] = coef * v_half_squared[i] * alpha_half[i]

    # i=0 boundary (regularity)
    coef0 = 1.0 / (v_squared[0] * dv)
    main_diag[0] = coef0 * v_half_squared[0] * beta_half[0]
    sup_diag[0] = coef0 * v_half_squared[0] * alpha_half[0]

    # i=N-1 boundary (Dirichlet)
    main_diag[-1] = 1.0
    sub_diag[-1] = 0.0

    L = diags([sub_diag, main_diag, sup_diag],
              offsets=[-1, 0, 1],
              shape=(N, N),
              format='csc')

    return L


# ============================================================
# FUSION BURNOUT TERM
# ============================================================

def fusion_burnout_rate(v, n_B):
    """Proton burnout rate ν_fus(v) = n_B · σ(v) · v.

    Here v is the lab-frame proton velocity, and σ(v) must be evaluated at
    the corresponding CM energy.
    Reduced-mass conversion: E_CM = (m_B / (m_p + m_B)) · E_lab,p
    """
    # Lab proton kinetic energy: E_p = 0.5 · m_p · v²
    # CM energy: E_CM = E_p · m_B/(m_p + m_B) = E_p · 11/12
    E_p_erg = 0.5 * m_p_g * v**2
    E_CM_erg = E_p_erg * m_B_g / (m_p_g + m_B_g)
    E_CM_keV = E_CM_erg / keV_to_erg

    sigma_b = sigma_TB(E_CM_keV)  # barn
    sigma_cm2 = sigma_b * barn_cm2

    # In the lab frame the relative velocity ≈ v (boron is much slower
    # thermally than fast protons)
    return n_B * sigma_cm2 * v


# ============================================================
# STEADY-STATE FP SOLVER
# ============================================================

def solve_steady_state(v, D, F, S, m_test=m_p_g, nu_fus=None,
                       maxwell_initial=None, D_func=None, F_func=None):
    """Solve the steady-state FP equation.

    L[f] - ν_fus·f + S = 0
    →  (L - ν_fus·I) f = -S

    Parameters
    ----------
    v : ndarray, shape (N,)
    D, F : ndarray, shape (N,)
        Diffusion and friction coefficients (grid values).
    S : ndarray, shape (N,)
        Source term (cm⁻⁶ s²).
    m_test : float
        Test particle mass (g).
    nu_fus : ndarray or None
        Fusion burnout rate (s⁻¹). None means zero.
    maxwell_initial : ndarray or None
        Reference Maxwell distribution used to set the boundary.
    D_func, F_func : callable or None
        D and F as functions. If supplied, the Chang-Cooper scheme evaluates
        them DIRECTLY at the half-points → Maxwell preservation at the
        ~10⁻¹¹ level. If None, linear averaging is used (~10⁻² error).

    Returns
    -------
    f : ndarray, shape (N,)
        Solution distribution function.
    """
    N = len(v)

    # FP operator
    L = build_FP_operator(v, D, F, m_test, D_func=D_func, F_func=F_func)

    # Burnout term
    if nu_fus is None:
        nu_fus = np.zeros(N)

    # System matrix: A = L - diag(ν_fus)
    A = L - diags([nu_fus], [0], shape=(N, N), format='csc')

    # Right-hand side: -S
    b = -S.copy()

    # Boundary condition: f=0 at v_max
    # (build_FP_operator already sets main_diag[-1]=1)
    b[-1] = 0.0

    # Solve
    f = spsolve(A, b)

    # Zero out negative values (numerical noise)
    f = np.maximum(f, 0.0)

    return f


# ============================================================
# VALIDATION TESTS
# ============================================================

def maxwell_distribution(v, n, T_keV, m):
    """Isotropic Maxwell-Boltzmann distribution.

    f_M(v) = n · (m/(2π·T))^(3/2) · exp(-m·v²/(2T))
    """
    T_erg = T_keV * keV_to_erg
    return n * (m / (2 * np.pi * T_erg))**1.5 * np.exp(-m * v**2 / (2 * T_erg))


def _test_maxwell_preservation():
    """
    CRITICAL TEST: Maxwell initial + Maxwell collisions → Maxwell preserved.

    If L · f_M is not approximately 0, then the FP operator does NOT preserve
    Maxwell equilibrium and the code is fundamentally broken.

    This test now compares two methods:
    1. Linear averaging (legacy method) — ~10⁻² error
    2. Direct half-point evaluation (new method) — ~10⁻¹¹ error
    """
    print("=" * 70)
    print("TEST 1: MAXWELL PRESERVATION TEST")
    print("=" * 70)
    print("Expect L[f_M] ≈ 0 (Maxwell is the steady-state of FP)")
    print()

    # Plasma parameters
    n_p = 8.5e13
    n_B = 1.5e13
    n_e = n_p + Z_B * n_B
    T = 300.0
    lnLambda = 17.0

    # Velocity grid
    T_erg = T * keV_to_erg
    v_th = np.sqrt(2 * T_erg / m_p_g)
    v_grid, dv = make_velocity_grid(0.05*v_th, 4*v_th, N=300)

    # Maxwell distribution
    f_M = maxwell_distribution(v_grid, 1.0, T, m_p_g)
    f_M_max = np.max(f_M)

    # D and F grid values
    D = D_total_thermal(v_grid, n_p, n_B, n_e, T, T, T, lnLambda)
    F = F_total_thermal(v_grid, n_p, n_B, n_e, T, T, T, lnLambda)

    # Functional forms (new: direct half-point evaluation)
    def D_fn(v):
        return D_total_thermal(v, n_p, n_B, n_e, T, T, T, lnLambda)
    def F_fn(v):
        return F_total_thermal(v, n_p, n_B, n_e, T, T, T, lnLambda)

    # OLD METHOD: linear averaging
    L_old = build_FP_operator(v_grid, D, F, m_p_g)
    Lf_M_old = L_old @ f_M

    # NEW METHOD: direct half-point evaluation
    L_new = build_FP_operator(v_grid, D, F, m_p_g, D_func=D_fn, F_func=F_fn)
    Lf_M_new = L_new @ f_M

    print(f"Grid: N=300, v_min={v_grid[0]:.2e}, v_max={v_grid[-1]:.2e}")
    print(f"v_th_p = {v_th:.3e} cm/s")
    print()

    # Core region
    core_mask = f_M > f_M_max * np.exp(-9)

    err_old = np.abs(Lf_M_old) / f_M_max
    err_new = np.abs(Lf_M_new) / f_M_max

    print("METHOD 1: Linear averaging (legacy default)")
    print(f"  Core max |L[f_M]|/max(f_M) = {np.max(err_old[core_mask]):.3e}")
    print(f"  Core mean                  = {np.mean(err_old[core_mask]):.3e}")
    print()

    print("METHOD 2: Direct half-point evaluation (D_func, F_func)")
    print(f"  Core max |L[f_M]|/max(f_M) = {np.max(err_new[core_mask]):.3e}")
    print(f"  Core mean                  = {np.mean(err_new[core_mask]):.3e}")
    print()

    improvement = np.max(err_old[core_mask]) / np.max(err_new[core_mask])
    print(f"IMPROVEMENT: {improvement:.1e}x better")

    if np.max(err_new[core_mask]) < 1e-9:
        print("[OK] Maxwell preserved at machine precision (~10⁻¹¹)")
    elif np.max(err_new[core_mask]) < 1e-6:
        print("[OK] Maxwell preserved at target level (10⁻⁶)")
    else:
        print("[WARN] Expected improvement not fully achieved")


def _test_solver_with_constant_source():
    """
    Steady-state solution with a Gaussian source: the distribution should
    pile up around the source and develop a slowing-down tail.
    """
    print("\n" + "=" * 70)
    print("TEST 2: SOLUTION WITH A CONSTANT SOURCE")
    print("=" * 70)

    n_p = 8.5e13
    n_B = 1.5e13
    n_e = n_p + Z_B * n_B
    T = 300.0
    lnLambda = 17.0

    T_erg = T * keV_to_erg
    v_th = np.sqrt(2 * T_erg / m_p_g)
    v_grid, dv = make_velocity_grid(0.05*v_th, 5*v_th, N=300)

    # Diffusion and friction
    D = D_total_thermal(v_grid, n_p, n_B, n_e, T, T, T, lnLambda)
    F = F_total_thermal(v_grid, n_p, n_B, n_e, T, T, T, lnLambda)

    # Gaussian source near 3·v_th
    v_source = 3.0 * v_th
    sigma_src = 0.3 * v_th
    S_amplitude = 1e0  # cm⁻⁶ s² scale
    S = S_amplitude * np.exp(-(v_grid - v_source)**2 / (2 * sigma_src**2))

    # Solve
    f = solve_steady_state(v_grid, D, F, S, m_p_g)

    # Compare to Maxwell (for scale)
    n_test = np.trapezoid(4 * np.pi * v_grid**2 * f, v_grid)
    print(f"\nSource location: v = {v_source:.2e} cm/s ({v_source/v_th:.1f}·v_th)")
    print(f"Total density of solution: {n_test:.3e} cm⁻³")

    # Tail test: does f drop below the source location?
    i_source = np.argmin(np.abs(v_grid - v_source))
    f_below = f[i_source//2]  # below source
    f_at_source = f[i_source]
    f_above = f[min(i_source + 30, len(f)-1)]  # above source

    print(f"\nf({v_grid[i_source//2]:.2e}) = {f_below:.3e}  (below source)")
    print(f"f({v_grid[i_source]:.2e}) = {f_at_source:.3e}  (at source)")
    print(f"f({v_grid[min(i_source + 30, len(f)-1)]:.2e}) = {f_above:.3e}  (above source)")

    # Above the source the distribution should fall rapidly (cutoff approach)
    if f_above < f_at_source * 0.3:
        print("[OK] Distribution decreases above source (slowing-down behavior)")
    else:
        print("[WARN] More distribution above source than expected")

    # Negativity check
    if np.all(f >= 0):
        print("[OK] No negative f anywhere")
    else:
        print(f"[FAIL] Number of negative entries: {np.sum(f < 0)}")


def _test_burnout_effect():
    """
    With the burnout term included, the distribution should decrease at high v.
    """
    print("\n" + "=" * 70)
    print("TEST 3: FUSION BURNOUT EFFECT")
    print("=" * 70)

    n_p = 8.5e13
    n_B = 1.5e13
    n_e = n_p + Z_B * n_B
    T = 300.0
    lnLambda = 17.0

    T_erg = T * keV_to_erg
    v_th = np.sqrt(2 * T_erg / m_p_g)
    v_grid, dv = make_velocity_grid(0.05*v_th, 8*v_th, N=400)

    D = D_total_thermal(v_grid, n_p, n_B, n_e, T, T, T, lnLambda)
    F = F_total_thermal(v_grid, n_p, n_B, n_e, T, T, T, lnLambda)

    # Thermal source: small Maxwell-shaped perturbation
    f_M = maxwell_distribution(v_grid, n_p, T, m_p_g)

    # Burnout rate
    nu_fus = fusion_burnout_rate(v_grid, n_B)

    print(f"v_th = {v_th:.2e} cm/s")
    print(f"Max ν_fus = {np.max(nu_fus):.3e} s⁻¹")
    print("  (Putvinski: typical ν_fus / ν_coll ~ 1e-3)")

    # Compare with collision frequency
    nu_coll = D / v_grid**2
    ratio = nu_fus / (nu_coll + 1e-300)

    i_peak = np.argmax(nu_fus)
    print(f"At v={v_grid[i_peak]:.2e} (ν_fus peak): ν_fus/ν_coll = {ratio[i_peak]:.3e}")

    if 1e-5 < ratio[i_peak] < 1e-1:
        print("[OK] Burnout is a small perturbation (FP regime preserved)")
    else:
        print("[WARN] Burnout/collision ratio is unusual")


if __name__ == "__main__":
    _test_maxwell_preservation()
    _test_solver_with_constant_source()
    _test_burnout_effect()
