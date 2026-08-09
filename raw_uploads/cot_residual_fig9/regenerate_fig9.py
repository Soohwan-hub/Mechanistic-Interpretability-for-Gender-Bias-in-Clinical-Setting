import os, json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_figures")

# Flipped ramp: green negative -> #f1f1f1 zero -> orange positive.
COLORSCALE = [
    [0.0, "#488f31"], [1/6, "#6aaa96"], [2/6, "#aecdc2"],
    [0.5, "#f1f1f1"],
    [4/6, "#f8b9a1"], [5/6, "#f08056"], [1.0, "#de3e00"],
]
TICK_FONT, AXIS_TITLE_FONT, TITLE_FONT = 12, 15, 20
N_TOK = 15
CONDITIONS = ["depression", "multiple sclerosis", "rheumatoid arthritis", "sarcoidosis"]

# (prompt_dir, variant, leftmost/highest token position) verified label-for-label
# against the source figures. Note MS uses PROMPT_C; the others use PROMPT_A.
SPEC = {
    "var1": {
        "depression":           ("VIGNETTE_PROMPT_A", "var1", 156),
        "multiple sclerosis":   ("VIGNETTE_PROMPT_C", "var1", 147),
        "rheumatoid arthritis": ("VIGNETTE_PROMPT_A", "var1", 162),
        "sarcoidosis":          ("VIGNETTE_PROMPT_A", "var1", 160),
    },
    "var2": {
        "depression":           ("VIGNETTE_PROMPT_A", "var2", 153),
        "multiple sclerosis":   ("VIGNETTE_PROMPT_C", "var2", 150),
        "rheumatoid arthritis": ("VIGNETTE_PROMPT_A", "var2", 159),
        "sarcoidosis":          ("VIGNETTE_PROMPT_A", "var2", 157),
    },
}


def clean(t):
    s = t.strip()
    return "\\n" if s == "" else s


def build(figvar, out_name, title):
    panels = []
    for c in CONDITIONS:
        pr, v, top = SPEC[figvar][c]
        base = f"{c.replace(' ', '_')}_{pr[-1]}_{v}"
        M = np.loadtxt(os.path.join(DATA, base + "_rewrite_matrix.csv"), delimiter=",")
        tl = json.load(open(os.path.join(DATA, base + "_token_labels.json")))
        cols = [top - k for k in range(N_TOK)]          # descending, as in the source
        sub = M[:, cols]
        toks = [f"{clean(tl[i])}_{i}" for i in cols]
        panels.append((c, sub, toks, list(range(M.shape[0]))))
        print(f"  {c:22s} {pr[-1]}/{v:6s} pos {cols[-1]}..{top}  "
              f"L0-21 max={np.abs(sub[:22]).max():.3f}  L22+ max={np.abs(sub[22:]).max():.4f}")

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.045,
                        subplot_titles=[c for c in CONDITIONS])
    for i, (c, z, toks, layers) in enumerate(panels, start=1):
        fig.add_trace(go.Heatmap(
            z=z, x=toks, y=layers, colorscale=COLORSCALE,
            zmid=0, zmin=-1, zmax=1, xgap=1, ygap=1,
            showscale=(i == len(panels)),
            colorbar=dict(title=dict(text="Rewrite Score", side="right",
                                     font=dict(size=AXIS_TITLE_FONT)),
                          len=0.82, thickness=16, x=1.012,
                          tickvals=[-1, -0.5, 0, 0.5, 1],
                          tickfont=dict(size=TICK_FONT)),
        ), row=1, col=i)
        fig.update_yaxes(title="Layer" if i == 1 else None,
                         title_font=dict(size=AXIS_TITLE_FONT),
                         tickmode="array", tickvals=layers,
                         ticktext=[str(l) for l in layers],
                         autorange="reversed", tickfont=dict(size=TICK_FONT - 2),
                         showticklabels=(i == 1), automargin=True, row=1, col=i)
        fig.update_xaxes(title="Token", title_font=dict(size=AXIS_TITLE_FONT),
                         tickangle=60, tickfont=dict(size=TICK_FONT - 2),
                         automargin=True, row=1, col=i)

    fig.update_layout(title=dict(text=title, font=dict(size=TITLE_FONT), x=0.015, xanchor="left"),
                      plot_bgcolor="white", paper_bgcolor="white",
                      width=1700, height=780, margin=dict(t=105, b=190, l=75, r=150))
    for ann in fig.layout.annotations:
        if ann.text in CONDITIONS:
            ann.font.size = AXIS_TITLE_FONT + 1
    p = os.path.join(OUT, out_name)
    fig.write_image(p, scale=3)
    print("Wrote:", p)


print("var1:")
build("var1", "fig9_cot_residual_by_condition_promptA_var1.png",
      "CoT Residual-Stream Rewrite Score by Condition (Qwen, var1)")
print("var2:")
build("var2", "fig9b_cot_residual_by_condition_promptA_var2.png",
      "CoT Residual-Stream Rewrite Score by Condition (Qwen, var2)")
