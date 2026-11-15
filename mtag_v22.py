"""v2.2 endcap likelihood material tag (preregistered).  Extension of the v2.1 barrel tag to
forward rapidities (1.5 < |eta| < 2.5, endcap disk geometry).  The primary material-proximity
coordinate is z_m (measured vertex z) instead of r_m; the tag formula is otherwise identical.
  p_conv(z) = sum_d c_d * [Phi((z - Z_d + HW_d)/s) - Phi((z - Z_d - HW_d)/s)] / (2 HW_d),
              c_d proportional to X0_d, sum c_d = 1
  p_sig(z)  = 1 / (Z_HI - Z_LO) mm^{-1}  (flat, signal-agnostic)
  lambda    = dchi2 / 2 + ln[p_sig(z) / p_conv(z)]          (pairs without valid dchi2 use dchi2 = 0)

Disk positions and material budgets are motivated by the CMS Phase-2 Inner Tracker endcap
(TFPX: 8 small disks; TEPX: 4 large disks) and Outer Tracker endcap (TEDD), simplified for
this toy.  See the preregistration README for details."""
import numpy as np
from scipy.special import ndtr

# Endcap disk material bands: (z_position [mm], X0_fraction at f = 1, half_width_z [mm])
#   — 9 bands covering z = 230–2300 mm
#   — inner pixel disks (TFPX-motivated): moderate X0, narrow half-width
#   — outer disks (TEPX/TEDD-motivated): higher X0, wider half-width
#   — transition band at z = 230 mm: services, cables, cooling at barrel-endcap junction
DISKS = [(230, 0.006, 1.5), (300, 0.020, 3.0), (450, 0.020, 3.0), (620, 0.020, 3.0),
         (820, 0.022, 3.5), (1100, 0.030, 4.5), (1500, 0.030, 5.0),
         (1900, 0.035, 5.5), (2300, 0.035, 6.0)]
Z_LO, Z_HI = 200.0, 2500.0       # fiducial z range for flat signal prior [mm]
P_FLOOR = 1e-300

# Oblique incidence reference: particles cross disks at polar angle theta,
# effective material X_eff = X0 / cos(theta); cos(theta) = tanh(eta).
# The simulation uses eta_ref = 2.0 (central endcap) unless scanning eta.
ETA_REF = 2.0
COS_THETA_REF = float(np.tanh(ETA_REF))     # ~ 0.964


def p_conv(z, s, disks=DISKS):
    """Normalised conversion probability density at vertex z, given resolution s = sigma_z."""
    Z = np.array([d[0] for d in disks], float)
    X = np.array([d[1] for d in disks], float)
    H = np.array([d[2] for d in disks], float)
    c = X / X.sum()
    z = np.asarray(z, float)[..., None]
    s = np.maximum(np.asarray(s, float), 1e-6)[..., None]
    return (c * (ndtr((z - Z + H) / s) - ndtr((z - Z - H) / s)) / (2 * H)).sum(-1)


def p_sig(z):
    """Flat signal prior over fiducial z range."""
    return np.full(np.shape(z), 1.0 / (Z_HI - Z_LO))


def lam(z, s, dchi2):
    """Endcap likelihood material tag lambda(z_m, sigma_z, Delta chi^2_3D)."""
    d = np.where(np.isfinite(dchi2), dchi2, 0.0)
    return 0.5 * d + np.log(p_sig(z)) - np.log(np.maximum(p_conv(z, s), P_FLOOR))


TGRID = (None, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 25.0)
