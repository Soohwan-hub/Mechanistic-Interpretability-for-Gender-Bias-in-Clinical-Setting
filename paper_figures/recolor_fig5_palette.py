"""Recolor the fig5 combined heatmap at the pixel level.

Usage:
    python recolor_fig5_palette.py <source.png> <dest.png>

The committed _redpink.png was produced from the UNFILTERED fig5 render
(commit f0a910e), not from the current HEAD version, which carries the
L3-L21 / top-18-token display filtering added in 9b0a178. To reproduce it:

    git show f0a910e:paper_figures/fig5_mlp_heatmaps_combined_asthma_ra.png > /tmp/src.png
    python recolor_fig5_palette.py /tmp/src.png \
        paper_figures/fig5_mlp_heatmaps_combined_asthma_ra_redpink.png


Nothing is re-plotted: the source PNG's layout, tokens, values, fonts and
gridlines are preserved bit-for-bit. Each heatmap pixel's colour is inverted
back to its position t in [-1,1] on the ORIGINAL green/orange ramp, then
re-emitted through the new red/pink ramp. Pixels that don't lie on the original
ramp (text, axes, white background) are left untouched.
"""
import sys
import numpy as np
from PIL import Image

SRC, DST = sys.argv[1], sys.argv[2]

# Original plotly ramp used by fig5: orange(-1) -> #f1f1f1(0) -> green(+1)
OLD = [(0.0,"#de3e00"),(1/6,"#f08056"),(2/6,"#f8b9a1"),(0.5,"#f1f1f1"),
       (4/6,"#aecdc2"),(5/6,"#6aaa96"),(1.0,"#488f31")]
# New ramp, same structure: dark red(-1) -> #f1f1f1(0) -> pink(+1)
NEW = [(0.0,"#8f0707"),(1/6,"#be6c65"),(2/6,"#d3a8a4"),(0.5,"#f1f1f1"),
       (4/6,"#d7c2c1"),(5/6,"#de99a1"),(1.0,"#de6e8c")]

def hex2rgb(h):
    h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))

def ramp(stops, n=2001):
    """Densely sample a plotly-style piecewise-linear colour ramp."""
    pos=np.array([p for p,_ in stops]); cols=np.array([hex2rgb(c) for _,c in stops],float)
    t=np.linspace(0,1,n)
    out=np.zeros((n,3))
    for k in range(3):
        out[:,k]=np.interp(t,pos,cols[:,k])
    return t,out

t_old,C_old = ramp(OLD)
t_new,C_new = ramp(NEW)

im=Image.open(SRC).convert("RGB")
a=np.array(im).astype(np.int16)
h,w,_=a.shape
flat=a.reshape(-1,3)

# Map each unique colour once -- far faster than per-pixel, and exact.
uniq,inv=np.unique(flat,axis=0,return_inverse=True)
print(f"{SRC}: {w}x{h}, {len(uniq)} unique colours")

# Nearest point on the original ramp, in RGB space.
d=np.linalg.norm(uniq[:,None,:].astype(float)-C_old[None,:,:],axis=2)
idx=np.argmin(d,axis=1)
dist=d[np.arange(len(uniq)),idx]

# Only recolour what genuinely sits on the original ramp. Text (#2a3f5f),
# pure white and antialiased edges fall outside this tolerance and survive.
TOL=10.0
on=dist<=TOL
lut=uniq.copy()
lut[on]=np.rint(C_new[idx[on]]).astype(np.int16)
print(f"  recoloured {on.sum()} / {len(uniq)} unique colours")

out=lut[inv].reshape(h,w,3).astype(np.uint8)
Image.fromarray(out).save(DST)
print("wrote",DST)
