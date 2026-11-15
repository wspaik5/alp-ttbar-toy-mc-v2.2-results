"""v2.2 post-registration: endcap displaced ALP → e⁺e⁻ simulation.

Adapts the barrel framework (bkg.py / mag.py) to the CMS Phase-2 endcap disk geometry:
  - Conversion background generated at endcap disk z positions
  - ALP signal with forward-boosted kinematics at η_ref = 2.0
  - z_m replaces r_m as the primary material-proximity coordinate
  - Oblique incidence: X_eff = X₀ / tanh(η)
  - Opening-angle compression by cosh(η) at forward rapidity

Configurations (from preregistration):
  E1: f = 1.0, z precision from inner pixel disks only (300, 450, 620 mm)
  E2: f = 1.0, z precision from all disks
  E3: f = 0.25, z precision from all disks
"""
import numpy as np
import sys, os

# ── Import the frozen endcap tag definition ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'v22_prereg'))
import mtag_v22

# ── Constants ──
ME = 0.000511          # electron mass [GeV]
LUMI = 500.0e3         # pb⁻¹ (500 fb⁻¹)
F_ENDCAP = 0.20        # P(1.5 < |η| < 2.5) for η ~ N(0, 1.6), approximate

# ── Material from frozen mtag_v22 ──
DISKS = mtag_v22.DISKS   # [(z_mm, X0_frac, HW_mm), ...]
Z_LO, Z_HI = mtag_v22.Z_LO, mtag_v22.Z_HI
ETA_REF = mtag_v22.ETA_REF
COS_THETA_REF = mtag_v22.COS_THETA_REF

# Material scale from environment (like barrel V16_F)
F_MAT = float(os.environ.get('V16_F', '1'))
B_FIELD = float(os.environ.get('V16_B', '3.8'))

# Disk arrays (scaled by f)
DISK_Z = np.array([d[0] for d in DISKS], float)
DISK_X0 = np.array([d[1] for d in DISKS], float) * F_MAT
DISK_HW = np.array([d[2] for d in DISKS], float)

# Inner pixel disks (for E1 config: z precision only at these)
INNER_PIXEL_Z = np.array([300.0, 450.0, 620.0])

# Oblique incidence factor at η_ref
OBLIQUE = 1.0 / np.tanh(ETA_REF)     # ~1.037 at η=2.0

# ── Toy detector assumptions (adapted from barrel bkg.py) ──
SIGMA_TTGAMMA_PB = 1.0     # σ(tt̄+γ, pT>20 GeV) [pb]
ANG_RES_PIXEL = 0.3e-3     # rad
ANG_RES_STRIP = 0.8e-3     # rad
MS_TERM = 2.0e-3 * np.sqrt(F_MAT)   # rad⋅GeV, scaled by √f
P_RES = 0.02 * (3.8 / B_FIELD)      # relative momentum resolution
VTX_RES_PIXEL = 1.0        # mm, z vertex resolution (pixel-quality z)
VTX_RES_STRIP = 3.0        # mm, z vertex resolution (strip-quality or no z)
VTX_FLOOR = 0.05           # mm, minimum z resolution

# ── Sample sizes ──
N_CONV = 100_000
N_SIG = 60_000

# NSIG for material veto scan (same as barrel)
NSIG = (None, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)

# KGRID for exemption threshold scan (same as barrel)
KGRID = (None, 1, 2, 4, 9, 16, 25, 49)


def _unit(v):
    return v / np.linalg.norm(v, axis=1)[:, None]


def endcap_conversions(n=1_500_000, seed=5, eta_ref=ETA_REF, config='E2'):
    """Generate photon conversions at endcap disk z positions.

    Each photon traverses disks sequentially; conversion probability per disk
    is proportional to X_eff = X₀ / tanh(η) (oblique incidence).
    """
    rng = np.random.default_rng(seed)
    oblique = 1.0 / np.tanh(eta_ref)

    # Photon pT spectrum from tt̄ (same as barrel)
    pt = 20 * rng.uniform(0, 1, n)**-0.5
    # Rapidity: fixed at eta_ref (preregistration Sec. 3.7)
    eta = np.full(n, eta_ref)
    phi = rng.uniform(0, 2 * np.pi, n)

    # Sequential conversion through disks
    X_eff = DISK_X0 * oblique   # effective material per disk
    pconv = 1 - np.exp(-7.0 / 9.0 * X_eff[None, :])   # P(convert) per disk, shape (n, n_disks)
    # Broadcast to (n, n_disks)
    pconv = np.broadcast_to(pconv, (n, len(DISK_Z)))
    surv = np.cumprod(np.hstack([np.ones((n, 1)), 1 - pconv[:, :-1]]), axis=1)
    prob = surv * pconv
    cum = np.cumsum(prob, axis=1)
    u = rng.uniform(0, 1, n)
    disk_idx = (u[:, None] > cum).sum(axis=1)    # == len(DISK_Z) means no conversion
    conv = disk_idx < len(DISK_Z)

    pt, eta, phi, disk_idx = pt[conv], eta[conv], phi[conv], disk_idx[conv]
    n_conv = len(pt)

    # Vertex z: uniformly distributed within disk half-width
    z = DISK_Z[disk_idx] + rng.uniform(-1, 1, n_conv) * DISK_HW[disk_idx]

    # Vertex r: at η_ref, r = z / sinh(η_ref)
    r = z / np.sinh(eta_ref)

    # Energy sharing (Bethe–Heitler): dP/dx ~ 1 - 4/3 x(1-x)
    x = rng.uniform(0, 1, 4 * n_conv)
    acc = rng.uniform(0, 1, 4 * n_conv) < 1 - 4.0 / 3.0 * x * (1 - x)
    x = x[acc][:n_conv]

    # Photon direction in lab frame
    dirv = np.stack([np.cos(phi), np.sin(phi), np.sinh(eta)], 1) / np.cosh(eta)[:, None]
    p = pt * np.cosh(eta)
    e1 = dirv * (x * p)[:, None]
    e2 = dirv * ((1 - x) * p)[:, None]

    # Measure: smear directions and momenta, compute pair mass and vertex z resolution
    mm, zm, sz = _measure_endcap(z, r, e1, e2, rng, eta_ref, config)
    w = np.full(n_conv, SIGMA_TTGAMMA_PB * F_ENDCAP * LUMI / n)

    return dict(z=z, r=r, zm=zm, sz=sz, m=mm, w=w,
                base=_basic_cuts_endcap(z, e1, e2),
                vtx=np.stack([r * np.cos(phi), r * np.sin(phi), z], 1),
                e1=e1, e2=e2,
                n_photons=SIGMA_TTGAMMA_PB * F_ENDCAP * LUMI,
                frac_conv=float(conv.mean()),
                disk_idx=disk_idx)


def endcap_signal(m, ctau=1.0, n=300_000, seed=11, eta_ref=ETA_REF, config='E2'):
    """Generate ALP → e⁺e⁻ signal at forward rapidity η_ref.

    The ALP is produced with pT from tt̄ spectrum, at fixed η = η_ref.
    The decay vertex z is drawn from the exponential proper-lifetime distribution,
    z_vtx = (cτ/m) × p_z × τ, where τ ~ Exp(1).
    Opening angle compressed by cosh(η_ref) vs barrel.
    """
    rng = np.random.default_rng(seed)

    # ALP kinematics
    pt = 20 + rng.exponential(80, 3 * n)
    # Fixed η_ref (preregistration)
    eta = np.full(3 * n, eta_ref)
    keep = np.ones(3 * n, bool)
    pt, eta = pt[keep][:n], eta[keep][:n]
    n = len(pt)
    phi = rng.uniform(0, 2 * np.pi, n)

    p = pt * np.cosh(eta)
    E = np.sqrt(p**2 + m**2)

    # ALP four-momentum
    pa = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], 1)

    # Exponential decay: L = lab-frame decay length
    L = rng.exponential(p / m * ctau)

    # Vertex position
    vtx = pa / p[:, None] * L[:, None]
    z_vtx = vtx[:, 2]  # vertex z

    # Decay in rest frame
    pstar = np.sqrt(m**2 / 4 - ME**2)
    cth = rng.uniform(-1, 1, n)
    sth = np.sqrt(1 - cth**2)
    ph = rng.uniform(0, 2 * np.pi, n)
    d = np.stack([sth * np.cos(ph), sth * np.sin(ph), cth], 1)

    # Boost to lab frame
    bvec = pa / E[:, None]
    gam = E / m

    def boost(pv, e):
        bp = (bvec * pv).sum(1)
        return pv + (gam**2 / (gam + 1) * bp + gam * e)[:, None] * bvec

    Es = np.sqrt(pstar**2 + ME**2)
    e1 = boost(pstar * d, Es)      # e-
    e2 = boost(-pstar * d, Es)     # e+

    # Vertex r from z: r = z / sinh(η_ref)
    r_vtx = np.abs(z_vtx) / np.sinh(eta_ref)

    # Measure
    mm, zm, sz = _measure_endcap(z_vtx, r_vtx, e1, e2, rng, eta_ref, config)
    w = np.full(n, F_ENDCAP * LUMI / n)

    return dict(z=z_vtx, r=r_vtx, zm=zm, sz=sz, m=mm, w=w,
                base=_basic_cuts_endcap(z_vtx, e1, e2),
                vtx=vtx, e1=e1, e2=e2)


def _measure_endcap(z_true, r_true, e1, e2, rng, eta_ref, config):
    """Smear the vertex z and momenta; return (pair mass, z_measured, sigma_z).

    config controls z resolution:
      E1: pixel-quality z only when vertex is near inner pixel disks (300, 450, 620 mm)
      E2/E3: pixel-quality z at all disks
    """
    # Determine if vertex is near a disk with z measurement
    if config == 'E1':
        # Only inner pixel disks provide z precision
        near_pixel = np.zeros(len(z_true), bool)
        for zd in INNER_PIXEL_Z:
            near_pixel |= (np.abs(z_true - zd) < 50.0)   # within 50 mm
        sz_base = np.where(near_pixel, VTX_RES_PIXEL, VTX_RES_STRIP)
    else:
        # E2, E3: all disks provide z precision
        near_any = np.zeros(len(z_true), bool)
        for zd in DISK_Z:
            near_any |= (np.abs(z_true - zd) < 50.0)
        sz_base = np.where(near_any, VTX_RES_PIXEL, VTX_RES_STRIP)

    # Kinematic-dependent z resolution (adapted from barrel vtx.py)
    # Forward boost compresses opening angle: α_T ~ 2m/(pT cosh η)
    pt1 = np.hypot(e1[:, 0], e1[:, 1])
    pt2 = np.hypot(e2[:, 0], e2[:, 1])
    p1 = np.linalg.norm(e1, axis=1)
    p2 = np.linalg.norm(e2, axis=1)

    # Transverse opening angle
    dphi = np.arctan2(e1[:, 1], e1[:, 0]) - np.arctan2(e2[:, 1], e2[:, 0])
    alpha_T = np.abs(np.angle(np.exp(1j * dphi)))

    # Track direction resolution
    pix = sz_base < 2.0   # use pixel resolution where vertex is pixel-quality
    sig_phi = np.sqrt(np.where(pix, ANG_RES_PIXEL, ANG_RES_STRIP)**2 +
                      (MS_TERM / np.minimum(p1, p2))**2)

    # Magnetic field separation (tangent point)
    k_mag = 0.3 * B_FIELD / 1000.0 * (1.0 / pt1 + 1.0 / pt2)   # 1/mm

    # Free-vertex and tangent-point resolution (barrel vtx.py model)
    hit_res = np.where(pix, 0.015, 0.035)   # mm
    gap = 10.0  # approximate gap to first measurement layer, mm
    pos_err = np.sqrt(2) * np.sqrt(hit_res**2 + gap**2 * sig_phi**2)
    with np.errstate(divide='ignore'):
        s_free = np.where(alpha_T > 0, pos_err / alpha_T, np.inf)
    s_tan = np.sqrt(2) * sig_phi / k_mag

    # Combined resolution
    s_kin = 1.0 / np.sqrt(1.0 / s_free**2 + 1.0 / s_tan**2)
    sz = np.hypot(np.minimum(s_kin, sz_base), VTX_FLOOR)

    # Smear z
    zm = z_true + rng.normal(0, 1, len(z_true)) * sz

    # Smear momenta and compute pair mass (same as barrel)
    out = []
    for e in (e1, e2):
        p = np.linalg.norm(e, axis=1)
        s = np.sqrt(np.where(pix, ANG_RES_PIXEL, ANG_RES_STRIP)**2 + (MS_TERM / p)**2)
        u = _unit(e)
        a = _unit(np.cross(u, np.array([0, 0, 1.0])[None, :] + 1e-9))
        b = np.cross(u, a)
        u2 = _unit(u + a * rng.normal(0, 1, len(p))[:, None] * s[:, None]
                    + b * rng.normal(0, 1, len(p))[:, None] * s[:, None])
        out.append(u2 * (p * (1 + P_RES * rng.normal(0, 1, len(p))))[:, None])
    q1, q2 = out
    E1_m = np.linalg.norm(q1, axis=1)
    E2_m = np.linalg.norm(q2, axis=1)
    m2 = (E1_m + E2_m)**2 - ((q1 + q2)**2).sum(1)
    mm = np.sqrt(np.clip(m2, 0, None))

    return mm, zm, sz


def _basic_cuts_endcap(z, e1, e2):
    """Endcap base selection: fiducial z range, minimum pT."""
    pt1 = np.hypot(e1[:, 0], e1[:, 1])
    pt2 = np.hypot(e2[:, 0], e2[:, 1])
    return ((z > Z_LO) & (z < Z_HI) &
            (np.minimum(pt1, pt2) > 1.0) &
            (pt1 + pt2 > 20))


def endcap_material_veto(z_meas, sig_z=None, nsig=2.0):
    """True if measured vertex z is AWAY from every disk band (passes the veto).

    Band at Z_d with half-width HW_d is widened by nsig × σ_z.
    """
    if sig_z is None:
        sig_z = np.full(len(z_meas), VTX_RES_PIXEL)
    d = np.abs(z_meas[:, None] - DISK_Z[None, :]) - DISK_HW[None, :]
    return (d > nsig * sig_z[:, None]).all(axis=1)


def upper_limit(S_per_pb, B):
    """Expected 95% CL limit: n_up = max(3, 2√B) / S."""
    n_up = max(3.0, 2.0 * np.sqrt(B))
    return n_up / S_per_pb if S_per_pb > 0 else np.inf


def subsample(d, nmax):
    """First nmax base pairs; same as barrel run_v16.subsample."""
    idx = np.where(d['base'])[0]
    use = idx[:nmax]
    out = dict(d)
    base = np.zeros(len(d['base']), bool)
    base[use] = True
    out['base'] = base
    out['w'] = d['w'] * (len(idx) / len(use))
    return out, use


def compute_dchi2_endcap(sig_or_conv, eta_ref=ETA_REF, seed_T=31, seed_L=1031):
    """Compute Δχ²_3D = Δχ²_T + Δχ²_L for endcap pairs.

    For this toy, Δχ²_T tests the transverse opening angle:
      - Conversion hypothesis: opening angle ~ 0 (collinear)
      - ALP hypothesis: opening angle ~ 2m/(pT cosh η)
    The forward boost at η_ref compresses the opening angle by cosh(η_ref).

    Δχ²_L is the longitudinal fit component (z consistency).
    """
    rng_T = np.random.default_rng(seed_T)
    rng_L = np.random.default_rng(seed_L)

    e1 = sig_or_conv['e1']
    e2 = sig_or_conv['e2']
    n = len(e1)

    # Transverse opening angle
    dphi = np.arctan2(e1[:, 1], e1[:, 0]) - np.arctan2(e2[:, 1], e2[:, 0])
    alpha_T = np.abs(np.angle(np.exp(1j * dphi)))

    # Track momenta
    pt1 = np.hypot(e1[:, 0], e1[:, 1])
    pt2 = np.hypot(e2[:, 0], e2[:, 1])
    p1 = np.linalg.norm(e1, axis=1)
    p2 = np.linalg.norm(e2, axis=1)

    # Angular resolution
    sig_alpha = np.sqrt(2) * np.sqrt(ANG_RES_PIXEL**2 + (MS_TERM / np.minimum(p1, p2))**2)

    # Δχ²_T: conversion hypothesis has α=0; the ALP hypothesis doesn't constrain α.
    # Δχ²_T = (α_T / σ_α)² when α_T is well-measured; smaller Δχ² = more conversion-like
    dchi2_T = (alpha_T / sig_alpha)**2

    # Add measurement noise to make it realistic
    noise_T = rng_T.normal(0, 1, n)
    dchi2_T_meas = np.maximum(0, dchi2_T + 2 * np.sqrt(np.maximum(dchi2_T, 1)) * noise_T)

    # Δχ²_L: z-based longitudinal fit quality
    # For conversions at disk z, the z measurement is consistent with the disk (Δχ²_L ~ 0)
    # For signal, vertex z is random, potentially inconsistent with any disk
    z = sig_or_conv['z']
    sz = sig_or_conv['sz']

    # Nearest disk distance normalised by resolution
    d_to_disk = np.abs(z[:, None] - DISK_Z[None, :]) / sz[:, None]
    min_dist = d_to_disk.min(axis=1)

    # Δχ²_L: pulls from nearest disk (conversion-like vertex near disk has small Δχ²_L)
    dchi2_L = min_dist**2
    noise_L = rng_L.normal(0, 1, n)
    dchi2_L_meas = np.maximum(0, dchi2_L + 2 * np.sqrt(np.maximum(dchi2_L, 1)) * noise_L)

    # Combined
    dchi2_3D = dchi2_T_meas + dchi2_L_meas

    # Fit success flag: true unless something pathological
    ok_alp = np.isfinite(dchi2_3D) & (p1 > 0.5) & (p2 > 0.5)

    return dchi2_3D, ok_alp


def scan_v15_endcap(m, sig, bkg_d, dchi_s, dchi_b, kgrid=KGRID, mass_window=True):
    """scan_v15 adapted for endcap: hard z-based material veto + exemption threshold k.

    Identical logic to barrel scan_v15 but using endcap_material_veto (z-based).
    """
    S0 = sig['w'][sig['base']].sum()
    lim0 = upper_limit(S0, 0)
    best = None

    cands = [(np.ones(len(sig['z']), bool), np.ones(len(bkg_d['z']), bool))]
    if mass_window:
        for lo in np.linspace(0, 1.5 * m, 31):
            for hi in (3 * m + 0.01, np.inf):
                cands.append(((sig['m'] > lo) & (sig['m'] < hi),
                              (bkg_d['m'] > lo) & (bkg_d['m'] < hi)))

    for ns in NSIG:
        vs0 = (np.ones(len(sig['z']), bool) if ns is None
               else endcap_material_veto(sig['zm'], sig['sz'], ns))
        vb0 = (np.ones(len(bkg_d['z']), bool) if ns is None
               else endcap_material_veto(bkg_d['zm'], bkg_d['sz'], ns))
        for kk in kgrid:
            if kk is None:
                vs, vb = vs0, vb0
            else:
                vs = vs0 | (dchi_s >= kk)
                vb = vb0 | (dchi_b >= kk)
            for ms, mb in cands:
                cs = sig['base'] & vs & ms
                cb = bkg_d['base'] & vb & mb
                S = sig['w'][cs].sum()
                B = bkg_d['w'][cb].sum()
                lim = upper_limit(S, B)
                if best is None or lim < best['limit']:
                    best = dict(limit=lim, eff=cs.mean() * 100, B=B,
                                nsig=ns, k=kk, deg=lim / lim0)
    return best


def scan_tag_endcap(m, sig, bkg_d, lam_sig, lam_conv, mass_window=True):
    """Scan the endcap likelihood tag threshold over TGRID × mass windows."""
    TGRID = mtag_v22.TGRID
    S0 = sig['w'][sig['base']].sum()
    lim0 = upper_limit(S0, 0)
    best = None

    cands = [(np.ones(len(sig['z']), bool), np.ones(len(bkg_d['z']), bool))]
    if mass_window:
        for lo in np.linspace(0, 1.5 * m, 31):
            for hi in (3 * m + 0.01, np.inf):
                cands.append(((sig['m'] > lo) & (sig['m'] < hi),
                              (bkg_d['m'] > lo) & (bkg_d['m'] < hi)))

    for t in TGRID:
        if t is None:
            ts = np.ones(len(sig['z']), bool)
            tb = np.ones(len(bkg_d['z']), bool)
        else:
            ts = lam_sig >= t
            tb = lam_conv >= t
        for ms, mb in cands:
            cs = sig['base'] & ts & ms
            cb = bkg_d['base'] & tb & mb
            S = sig['w'][cs].sum()
            B = bkg_d['w'][cb].sum()
            lim = upper_limit(S, B)
            if best is None or lim < best['limit']:
                best = dict(limit=lim, deg=lim / lim0, t=t,
                            eff_sig=float(cs.mean() * 100), B=float(B), S=float(S))
    return best
