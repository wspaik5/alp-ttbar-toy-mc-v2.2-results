"""Generate figures for v2.2 post-registration report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'v22_prereg'))
import mtag_v22

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Load results ──
res = np.load(os.path.join(OUT, 'v22_results.npy'), allow_pickle=True).item()


def fig_integration_failure():
    """Fig R1: Integration failure demonstration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    demo = res['integration_failure']
    sigmas = sorted(demo.keys())
    naive = [demo[s]['naive'] for s in sigmas]
    perband = [demo[s]['perband'] for s in sigmas]

    ax1.plot(sigmas, naive, 'rs-', label='Naïve quad (full range)', ms=8, lw=2)
    ax1.plot(sigmas, perband, 'go-', label='Per-band quad (±100 mm)', ms=8, lw=2)
    ax1.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5, label='Analytic = 1')
    ax1.set_xlabel(r'$\sigma_z$ [mm]', fontsize=12)
    ax1.set_ylabel(r'$\int p_{\mathrm{conv}}\,dz$', fontsize=12)
    ax1.set_title('(a) Integration of $p_{\\mathrm{conv}}$', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.4, 1.1)
    ax1.set_xticks(sigmas)

    # (b) p_conv shape at σ_z = 1 mm — show how narrow the peaks are
    z = np.linspace(200, 2500, 50000)
    pconv_1 = mtag_v22.p_conv(z, np.full_like(z, 1.0))
    pconv_15 = mtag_v22.p_conv(z, np.full_like(z, 15.0))

    ax2.plot(z, pconv_1, 'b-', lw=1.2, label=r'$\sigma_z = 1$ mm')
    ax2.plot(z, pconv_15, 'r-', lw=1.2, alpha=0.7, label=r'$\sigma_z = 15$ mm')

    for zd, x0, hw in mtag_v22.DISKS:
        ax2.axvline(zd, color='gray', ls=':', lw=0.5, alpha=0.5)

    ax2.set_xlabel('$z$ [mm]', fontsize=12)
    ax2.set_ylabel(r'$p_{\mathrm{conv}}(z)$ [mm$^{-1}$]', fontsize=12)
    ax2.set_title(r'(b) $p_{\mathrm{conv}}$ shape: 9 disk peaks', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(200, 2500)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_R1_integration_failure.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_degradation_summary():
    """Fig R2: Degradation summary for all configs."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    masses_MeV = [17, 50, 100, 200]
    cfgs = ['E1', 'E2', 'E3']
    colors_hv = '#2166ac'
    colors_tag = '#b2182b'

    for ax, cfg in zip(axes, cfgs):
        hv_degs = [res[cfg][('hv', m/1000)]['deg'] for m in masses_MeV]
        tag_degs = [res[cfg][('tag', m/1000)]['deg'] for m in masses_MeV]

        x = np.arange(len(masses_MeV))
        w = 0.35
        ax.bar(x - w/2, hv_degs, w, label='Hard veto', color=colors_hv, alpha=0.8)
        ax.bar(x + w/2, tag_degs, w, label='Tag', color=colors_tag, alpha=0.8)
        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}' for m in masses_MeV])
        ax.set_xlabel('$m_{a}$ [MeV]', fontsize=11)
        ax.set_title(f'Config {cfg}', fontsize=12)

    axes[0].set_ylabel('Degradation factor', fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle('Endcap degradation: hard veto vs. tag', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig_R2_degradation_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_barrel_vs_endcap():
    """Fig R3: Barrel (v2.1) vs endcap (v2.2) hard-veto degradation comparison."""
    # v2.1 barrel results (from summary)
    barrel = {
        ('C1', 17): 127.25, ('C1', 50): 63.88, ('C1', 100): 24.24, ('C1', 200): 6.09,
        ('C2', 17): 117.45, ('C2', 50): 38.54, ('C2', 100): 13.14, ('C2', 200): 3.93,
        ('C3', 17): 26.74,  ('C3', 50): 8.83,  ('C3', 100): 3.28,  ('C3', 200): 1.09,
    }
    masses_MeV = [17, 50, 100, 200]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    pairs = [('C1', 'E1'), ('C2', 'E2'), ('C3', 'E3')]
    for ax, (bc, ec_cfg) in zip(axes, pairs):
        barrel_vals = [barrel[(bc, m)] for m in masses_MeV]
        endcap_vals = [res[ec_cfg][('hv', m/1000)]['deg'] for m in masses_MeV]

        x = np.arange(len(masses_MeV))
        w = 0.35
        ax.bar(x - w/2, barrel_vals, w, label=f'Barrel {bc}', color='#4393c3', alpha=0.8)
        ax.bar(x + w/2, endcap_vals, w, label=f'Endcap {ec_cfg}', color='#d6604d', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}' for m in masses_MeV])
        ax.set_xlabel('$m_{a}$ [MeV]', fontsize=11)
        ax.set_title(f'{bc} vs {ec_cfg}', fontsize=12)
        ax.set_yscale('log')

    axes[0].set_ylabel('Hard-veto degradation', fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle('Barrel (v2.1) vs. endcap (v2.2): hard-veto degradation', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig_R3_barrel_vs_endcap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_endcap_geometry():
    """Fig R4: Endcap disk geometry — why the hard veto is so effective."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))

    for zd, x0, hw in mtag_v22.DISKS:
        # Disk band
        ax.fill_betweenx([0, x0 * 100], zd - hw, zd + hw,
                         color='#4393c3', alpha=0.5)
        # 4σ veto band (σ_z = 1 mm)
        sigma_z = 1.0
        veto_w = 4 * sigma_z + hw
        ax.fill_betweenx([0, x0 * 100 * 0.3], zd - veto_w, zd + veto_w,
                         color='#d6604d', alpha=0.2)

    ax.set_xlabel('$z$ [mm]', fontsize=12)
    ax.set_ylabel('$X_0$ [%]', fontsize=12)
    ax.set_title('Endcap disk bands (blue) and 4$\\sigma$ veto zones (red shading, $\\sigma_z=1$ mm)',
                 fontsize=11)
    ax.set_xlim(150, 2550)
    ax.set_ylim(0, 4.5)

    # Mark gaps
    disk_z = [d[0] for d in mtag_v22.DISKS]
    for i in range(len(disk_z) - 1):
        gap = disk_z[i+1] - disk_z[i]
        mid = (disk_z[i] + disk_z[i+1]) / 2
        ax.annotate(f'{gap:.0f} mm', xy=(mid, 4.0), ha='center', fontsize=7,
                    color='gray')

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_R4_endcap_geometry.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig_tag_utility():
    """Fig R5: Tag/hv ratio — barrel (v2.1) vs endcap (v2.2)."""
    # v2.1 barrel tag/hv ratios
    barrel_ratio = {
        ('C1', 17): 1.096, ('C1', 50): 1.033, ('C1', 100): 1.011, ('C1', 200): 1.007,
        ('C2', 17): 0.948, ('C2', 50): 1.012, ('C2', 100): 0.974, ('C2', 200): 0.979,
        ('C3', 17): 0.958, ('C3', 50): 0.989, ('C3', 100): 0.826, ('C3', 200): 0.997,
    }
    masses_MeV = [17, 50, 100, 200]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    pairs = [('C1', 'E1'), ('C2', 'E2'), ('C3', 'E3')]
    for ax, (bc, ec_cfg) in zip(axes, pairs):
        barrel_vals = [barrel_ratio[(bc, m)] for m in masses_MeV]
        endcap_vals = []
        for m in masses_MeV:
            hv_deg = res[ec_cfg][('hv', m/1000)]['deg']
            tag_deg = res[ec_cfg][('tag', m/1000)]['deg']
            endcap_vals.append(tag_deg / hv_deg)

        x = np.arange(len(masses_MeV))
        w = 0.35
        ax.bar(x - w/2, barrel_vals, w, label=f'Barrel {bc}', color='#4393c3', alpha=0.8)
        ax.bar(x + w/2, endcap_vals, w, label=f'Endcap {ec_cfg}', color='#d6604d', alpha=0.8)
        ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}' for m in masses_MeV])
        ax.set_xlabel('$m_{a}$ [MeV]', fontsize=11)
        ax.set_title(f'{bc} vs {ec_cfg}', fontsize=12)

        # Annotate: ratio < 1 = tag helps, > 1 = tag hurts
        ax.fill_between([-0.5, 3.5], 0, 1.0, color='#a6dba0', alpha=0.1)
        ax.fill_between([-0.5, 3.5], 1.0, max(max(barrel_vals), max(endcap_vals)) * 1.1,
                        color='#f4a582', alpha=0.1)
        ax.set_xlim(-0.5, 3.5)

    axes[0].set_ylabel('Tag / Hard-veto ratio', fontsize=11)
    axes[0].legend(fontsize=9, loc='upper right')

    # Add region labels to first axis
    axes[0].text(3.3, 0.5, 'Tag helps', fontsize=8, color='#1b7837', ha='right', style='italic')
    axes[0].text(3.3, max(barrel_ratio[('C1', 17)], 1.5), 'Tag hurts', fontsize=8,
                 color='#b2182b', ha='right', style='italic')

    fig.suptitle('Tag utility: tag/hard-veto ratio (barrel v2.1 vs. endcap v2.2)', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig_R5_tag_utility_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


if __name__ == '__main__':
    print('Generating v2.2 post-registration figures...')
    fig_integration_failure()
    fig_degradation_summary()
    fig_barrel_vs_endcap()
    fig_endcap_geometry()
    fig_tag_utility()
    print('Done.')
