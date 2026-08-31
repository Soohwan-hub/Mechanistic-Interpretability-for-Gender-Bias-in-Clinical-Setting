"""Regenerate paper Figures 1, 3, 5, 9 in the new divergent palette.

Each figure is produced by the ORIGINAL author's own plotting function --
the body is lifted out of the committed source with `ast` and executed as-is,
not retyped.  pio.write_image is intercepted so the palette is applied to the
figure that code already built.  Only colours change.

  Fig 1  female_condition_baseline_probe.py :: maybe_save_plots
  Fig 3  simple_patching.py                 :: plot_top_layers_bar
  Fig 5  recreate_simple_vs_cot_l18_plots.py:: plot_by_condition  (PALETTE dict)
  Fig 9  raw_uploads/cot_residual_fig9/regenerate_fig9.py (COLORSCALE only)

Two deviations from the committed code, both because the PAPER differs from
what was committed (the published figures came from tweaked off-repo copies):
  * Fig 1 legend reads Male/Female (committed: "Mean P(Male)"), sits top-left,
    white template, black bar borders.
  * Fig 3 y-axis reads "Score" (committed: "Rewrite score"), white template.

maybe_save_plots also emits fig1_gender_probs_by_prompt.png (not a
paper figure, not committed).

Figures 8 and 10 are NOT here -- no generator for Fig 8 exists on any branch,
and Fig 10's two-panel assembly / slice / label reformat are not committed.

Run from repo root:
    python figures_new_palette/regenerate.py figures_new_palette \
        <dir with simple_vs_cot_l18_comparison.csv> \
        <path to recreate_simple_vs_cot_l18_plots.py>

  the CSV lives on origin/sam/cot-behavioral-n35 under
  activation_patching/simple_patching/patching_results/qwen_cot20_mlp_rewrite/
  the Fig-5 module lives on origin/sam/cot-mlp-evidence
"""
import ast, json, pickle, sys, types
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as _real_pio

OUT = Path(sys.argv[1]); SAM = sys.argv[2]

# ---------------- the ONLY thing that is not Om's ----------------
P = dict(dark="#8f0707", mid="#be6c65", neutral="#d7c2c1",
         light="#de99a1", pink="#de6e8c")
DIV = [[0.0, P["dark"]], [0.25, P["mid"]], [0.5, P["neutral"]],
       [0.75, P["light"]], [1.0, P["pink"]]]
BAR_CYCLE = [P["dark"], P["pink"], P["mid"], P["light"]]
# -----------------------------------------------------------------

def extract(src_path, names):
    """Pull the named top-level functions out of a file, verbatim."""
    src = Path(src_path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = "".join(lines[node.lineno - 1: node.end_lineno])
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"could not extract {missing} from {src_path}")
    return out


def make_ns(dest: Path, only=None, post=None):
    """Namespace with a pio whose write_image recolours, then writes PNG."""
    captured = {}
    class PIO:
        @staticmethod
        def write_image(fig, out, *a, **k):
            d = dest
            if "by_prompt" in str(out):     # his fn emits a second figure; keep it too
                d = dest.with_name(dest.stem + "_by_prompt.png")
            # recolour whatever Om's code produced
            for tr in fig.data:
                t = tr.type
                if t == "bar":
                    tr.marker.color = BAR_CYCLE[captured.setdefault("n", 0) % 4]
                    captured["n"] = captured.get("n", 0) + 1
                elif t == "scatter":
                    i = captured.setdefault("s", 0); captured["s"] = i + 1
                    tr.line.color = BAR_CYCLE[i % 4]
                elif t in ("heatmap", "image"):
                    tr.colorscale = DIV
            fig.update_layout(coloraxis_colorscale=DIV)
            if post is not None:
                post(fig)
            _real_pio.write_image(fig, str(d), scale=3)
            print("   wrote", d.name)
    from statistics import mean as _mean
    ns = {"go": go, "px": px, "pio": PIO, "np": np, "pd": pd, "Path": Path,
          "mean": _mean, "_HAS_PLOTLY": True, "List": list, "Tuple": tuple,
          "Dict": dict, "Any": object, "Optional": object}
    return ns


REPO = Path(".")
SP  = REPO / "activation_patching/simple_patching/simple_patching.py"
FCB = REPO / "activation_patching/simple_patching/female_condition_baseline_probe.py"

# ---------- Fig 1 : Om's maybe_save_plots ----------
print("Fig 1  <- female_condition_baseline_probe.py :: maybe_save_plots")
rows = [json.loads(l) for l in
        open("activation_patching/simple_patching/female_bias_run1/female_condition_probs.jsonl")]
src = extract(FCB, ["maybe_save_plots"])["maybe_save_plots"]
def _fig1_paper(fig):
    for tr in fig.data:
        tr.name = tr.name.replace("Mean P(", "").replace(")", "")
        tr.marker.line = dict(color="black", width=1)     # paper: black bar borders
    fig.update_layout(legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
                      template="plotly_white")            # paper: white, not blue-grey
ns = make_ns(OUT / "fig1_gender_probs.png", post=_fig1_paper)
exec(src, ns)
ns["maybe_save_plots"](rows, OUT)          # writes by_condition then by_prompt
print(f"   rows={len(rows)}")

# ---------- Fig 3 : Om's plot_top_layers_bar ----------
print("Fig 3  <- simple_patching.py :: plot_top_layers_bar")
a = json.load(open("activation_patching/simple_patching/female5_patch_male/aggregate_per_layer.json"))
per, cnt = np.zeros(28), np.zeros(28)
for u in a["raw_units"]:
    for l, v in enumerate(u["score_stats"]["rewrite_scores"]["mean"]):
        per[l] += v; cnt[l] += 1
per /= np.maximum(cnt, 1)
src = extract(SP, ["plot_top_layers_bar"])["plot_top_layers_bar"]
def _fig3_paper(fig):
    for tr in fig.data:
        tr.marker.line = dict(color="black", width=1)      # black bar borders
    fig.update_yaxes(title_text="Score")
    fig.update_layout(template="plotly_white")            # paper: white, not blue-grey
ns = make_ns(OUT / "fig3_top_layers.png", post=_fig3_paper)
exec(src, ns)
ns["plot_top_layers_bar"]([(l, float(per[l])) for l in range(28)],
                          "Top layers (all conditions) by mean rewrite score",
                          str(OUT / "fig3_top_layers"), plot_format="png")

# ---------- Fig 5 : Om's module, PALETTE overridden ----------
print("Fig 5  <- recreate_simple_vs_cot_l18_plots.py :: plot_by_condition")
mod_src = Path(sys.argv[3]).read_text()
m5 = types.ModuleType("m5"); m5.__dict__["__name__"] = "m5"
exec(compile(mod_src, "recreate_simple_vs_cot_l18_plots.py", "exec"), m5.__dict__)
m5.PALETTE.update({"green_dark": P["dark"], "green_mid": P["mid"],
                   "green_light": P["neutral"], "neutral": P["neutral"],
                   "orange_light": P["neutral"], "orange_mid": P["light"],
                   "orange_dark": P["pink"]})
df = pd.read_csv(f"{SAM}/simple_vs_cot_l18_comparison.csv")
m5.plot_by_condition(df, OUT / "fig5_simple_vs_cot.png", dpi=160)
print("   wrote fig5_simple_vs_cot.png")


# ---------- Fig 9 : raw_uploads/cot_residual_fig9/regenerate_fig9.py ----------
print("Fig 9  <- regenerate_fig9.py :: COLORSCALE only")
import os as _os
f9 = Path("raw_uploads/cot_residual_fig9/regenerate_fig9.py")
src9 = f9.read_text()
old9 = '''COLORSCALE = [
    [0.0, "#488f31"], [1/6, "#6aaa96"], [2/6, "#aecdc2"],
    [0.5, "#f1f1f1"],
    [4/6, "#f8b9a1"], [5/6, "#f08056"], [1.0, "#de3e00"],
]'''
new9 = '''COLORSCALE = [
    [0.0, "#8f0707"], [0.25, "#be6c65"],
    [0.5, "#d7c2c1"],
    [0.75, "#de99a1"], [1.0, "#de6e8c"],
]'''
assert src9.count(old9) == 1, "fig9 COLORSCALE block not found verbatim"
src9 = src9.replace(old9, new9)
src9 = src9.replace(
    'DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")',
    f'DATA = {str(f9.parent / "data")!r}')
src9 = src9.replace(
    'OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_figures")',
    f'OUT = {str(OUT)!r}')
exec(compile(src9, str(f9), "exec"), {"__name__": "fig9", "os": _os})
_os.replace(OUT / "fig9_cot_residual_by_condition_promptA_var1.png",
            OUT / "fig9_cot_residual.png")
# his script also emits the var2 panel (paper fig 9b, not in this paper)
(OUT / "fig9b_cot_residual_by_condition_promptA_var2.png").unlink(missing_ok=True)
print("   wrote fig9_cot_residual.png")
