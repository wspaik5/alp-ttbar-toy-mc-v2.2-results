"""Build the v2.2 post-registration results PDF from README.md + 5 figures."""
import base64, hashlib, re, sys, os
import markdown
from playwright.sync_api import sync_playwright

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)

OUT = sys.argv[1]
src = open('README.md', encoding='utf-8').read()
sha = hashlib.sha256(open('README.md', 'rb').read()).hexdigest()

body_md = src.split('\n', 1)[1]                      # drop the H1 title
head, rest = body_md.split('\n---\n', 1)
head = '\n'.join(l + '  ' for l in head.strip().splitlines())

# GFM -> Python-Markdown fixes
lines, prev = [], ''
for l in rest.splitlines():
    if l.startswith('  - '):
        l = '    ' + l[2:]
    isl = lambda t: t.startswith('- ') or re.match(r'\d+\. ', t) is not None
    if isl(l) and prev.strip() and not isl(prev.lstrip()) and not prev.startswith('  '):
        lines.append('')
    lines.append(l); prev = l
rest = '\n'.join(lines)
html = markdown.markdown(head + '\n\n---\n' + rest, extensions=['tables'])

# [E1]..[E7] reference lines -> <br>
html = re.sub(r'\n(\[E\d\])', r'<br>\1', html)


def img(path, cap, w='100%'):
    b = base64.b64encode(open(path, 'rb').read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" style="width:{w}"><figcaption>{cap}</figcaption></figure>'


# ── Insert figures at their markdown placeholders ──
fig_map = {
    'fig_R1_integration_failure.png': img(
        'fig_R1_integration_failure.png',
        '<b>Fig R1.</b> (a) Integral of p<sub>conv</sub> via naïve full-range quad vs. per-band quad. '
        'At σ<sub>z</sub> = 1 mm, the naïve method misses ~46% of the probability mass. '
        '(b) Shape of p<sub>conv</sub> at σ<sub>z</sub> = 1 mm and 15 mm, showing the 9 disk peaks spread over 2300 mm.'),
    'fig_R2_degradation_summary.png': img(
        'fig_R2_degradation_summary.png',
        '<b>Fig R2.</b> Hard-veto and tag degradation for the three endcap configurations.', w='90%'),
    'fig_R3_barrel_vs_endcap.png': img(
        'fig_R3_barrel_vs_endcap.png',
        '<b>Fig R3.</b> Barrel (v2.1) vs. endcap (v2.2) hard-veto degradation (log scale).', w='90%'),
    'fig_R4_endcap_geometry.png': img(
        'fig_R4_endcap_geometry.png',
        '<b>Fig R4.</b> Endcap disk bands (blue) and 4σ veto zones (red shading, σ<sub>z</sub> = 1 mm). '
        'The large gaps between disks mean the veto covers ~6% of the fiducial z range.'),
    'fig_R5_tag_utility_comparison.png': img(
        'fig_R5_tag_utility_comparison.png',
        '<b>Fig R5.</b> Tag/hard-veto ratio for barrel (v2.1, blue) and endcap (v2.2, red). '
        'Green shading: tag helps (ratio &lt; 1); orange shading: tag hurts (ratio &gt; 1). '
        'The tag was already marginal in the barrel and becomes actively harmful in the endcap at low masses.', w='90%'),
}

# Replace markdown image syntax ![...](filename) with embedded figures
for fname, fig_html in fig_map.items():
    # Match the markdown-rendered <img> tag
    pattern = re.compile(r'<p><img [^>]*src="' + re.escape(fname) + r'"[^>]*/>\s*\n?<em>.*?</em></p>', re.DOTALL)
    if pattern.search(html):
        html = pattern.sub(fig_html, html, count=1)
    else:
        # Try simpler pattern
        pattern2 = re.compile(r'<img [^>]*src="' + re.escape(fname) + r'"[^>]*/>', re.DOTALL)
        html = pattern2.sub(fig_html, html, count=1)

PRE_DOI = '24189990'
front = f'''
<div class="title">ALP-ttbar Toy MC v2.2 — Post-Registration Results: Extension of the Likelihood Material Tag to
Forward Rapidities (Endcap Geometry) for Displaced Light ALP → e<sup>+</sup>e<sup>−</sup></div>
<div class="auth">Won Shik Paik · Independent Researcher, Auckland, New Zealand · 14 November 2026</div>
<div class="ids">Preregistration (frozen): <a href="https://doi.org/10.5281/zenodo.{PRE_DOI}">doi:10.5281/zenodo.{PRE_DOI}</a><br>
SHA-256 of this README.md: <code>{sha}</code></div>
<div class="box"><b>About this document.</b> This PDF reproduces the post-registration results README.md with embedded figures.
All computations were performed after the preregistration was published and its DOI was minted.
<b>Disclaimer.</b> Independent personal toy study, not affiliated with
the authors of arXiv:2610.38932 or arXiv:2306.08686, nor with the CMS, ATLAS, LHCb or ALICE Collaborations or the FCC
study. Detector numbers are assumed; only qualitative trends and ratios are meaningful.</div>'''

css = '''
@page { size: A4; }
body { font-family: "DejaVu Sans", sans-serif; font-size: 8.6pt; line-height: 1.38; color: #111; }
.title { font-size: 13pt; font-weight: bold; color: #12355b; line-height: 1.25; margin-bottom: 4px; }
.auth { font-size: 9pt; margin-bottom: 3px; } .ids { font-size: 7.8pt; color: #333; margin-bottom: 6px; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 7.4pt; }
.box { background: #f3f6fa; border-left: 3px solid #12355b; padding: 5px 8px; font-size: 7.9pt; margin: 6px 0 8px 0; }
h2 { font-size: 10.5pt; color: #12355b; border-bottom: 1px solid #c8d3e0; margin: 10px 0 4px 0; padding-bottom: 1px; }
h3 { font-size: 9.5pt; color: #12355b; margin: 8px 0 3px 0; }
p { margin: 3px 0; } ul, ol { margin: 2px 0 2px 0; padding-left: 18px; } li { margin: 1px 0; }
table { border-collapse: collapse; margin: 4px 0; font-size: 8pt; } th, td { border: 1px solid #c8d3e0; padding: 2px 6px; }
th { background: #eaf0f7; } hr { border: 0; border-top: 1px solid #c8d3e0; margin: 6px 0; }
figure { margin: 6px 0 8px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 7.6pt; text-align: left; color: #222; margin-top: 2px; }
a { color: #12355b; text-decoration: none; }
'''
doc = f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{front}{html}</body></html>'
open('results_v22.html', 'w', encoding='utf-8').write(doc)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.set_content(doc); pg.pdf(path=OUT, format='A4', print_background=True,
                                display_header_footer=True, header_template='<span></span>',
                                footer_template='<div style="font-size:7px;width:100%;text-align:center;color:#666">'
                                '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                                margin=dict(top='15mm', bottom='16mm', left='15mm', right='15mm'))
    b.close()
print('README sha256', sha)
print(f'PDF written to {OUT}')
