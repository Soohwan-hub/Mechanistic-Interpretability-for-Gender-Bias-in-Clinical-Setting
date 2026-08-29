"""
Regenerate RTAR paper figures.

Sources:
  Om's branch (om):              /tmp/om_branch/
  SAE artifacts (act_patch_simple): /tmp/act_patch_simple/
  CoT CSV (sam/cot-behavioral-n35): /tmp/sam_cot/

Run from repo root:
  cd /tmp/om_branch
  python paper_figures_out/generate_figures.py
"""

import sys, os, json, pickle, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
from figstyle import PALETTE, DARK, MID, NEUT, LIGHT, PINK, DIVERGING

OUT      = os.path.dirname(__file__)          # paper_figures_out/
REPO     = os.path.dirname(OUT)               # om branch root
ACT      = "/tmp/act_patch_simple"            # activation_patching_simple
SAE_DIR  = os.path.join(ACT,
    "activation_patching/cot_patching/sae_results/paper_figs_promptA_vs_C")
SAE_RUNS = os.path.join(ACT,
    "activation_patching/cot_patching/sae_results/sae_simple_runs")
SAM      = "/tmp/sam_cot"
COT_DIR  = os.path.join(SAM,
    "activation_patching/simple_patching/patching_results/qwen_cot20_mlp_rewrite")

warnings.filterwarnings("ignore")
SAVEKW = dict(dpi=300, bbox_inches="tight")

SKIPPED = []


def save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUT, f"{name}.{ext}"), **SAVEKW)
    print(f"  saved: {name}")
    plt.close("all")


def skip(name, reason):
    SKIPPED.append((name, reason))
    print(f"  SKIPPED {name}: {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# fig1_gender_probs_by_condition  /  fig1_gender_probs_by_condition_palette
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1_gender_probs():
    print("fig1_gender_probs_by_condition ...")
    path = os.path.join(REPO,
        "activation_patching/simple_patching/female_bias_run1/summary.json")
    if not os.path.exists(path):
        skip("fig1_gender_probs_by_condition", f"missing {path}")
        skip("fig1_gender_probs_by_condition_palette", f"missing {path}")
        return

    with open(path) as f:
        d = json.load(f)

    by_cond    = d["by_condition"]
    conditions = [r["condition"] for r in by_cond]
    p_female   = [r["mean_p_female"] for r in by_cond]
    p_male     = [r["mean_p_male"]   for r in by_cond]

    pretty = {
        "asthma": "Asthma",
        "bronchitis": "Bronchitis",
        "depression": "Depression",
        "essential hypertension": "Essential\nHypertension",
        "essential_hypertension": "Essential\nHypertension",
        "multiple_sclerosis": "Multiple\nSclerosis",
        "multiple sclerosis": "Multiple\nSclerosis",
        "rheumatoid_arthritis": "Rheumatoid\nArthritis",
        "rheumatoid arthritis": "Rheumatoid\nArthritis",
        "sarcoidosis": "Sarcoidosis",
    }
    xlabels = [pretty.get(c, c) for c in conditions]
    x, w    = np.arange(len(conditions)), 0.35

    for variant in ("base", "palette"):
        col_f, col_m = (DARK, LIGHT) if variant == "base" else (MID, NEUT)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - w/2, p_female, w, label="P(Female)", color=col_f)
        ax.bar(x + w/2, p_male,   w, label="P(Male)",   color=col_m)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel("Mean next-token probability", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Model gender prediction by clinical condition\n"
            "(Qwen2.5-7B, simple prompts, female→male patching)", fontsize=11)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        suffix = "" if variant == "base" else "_palette"
        save(f"fig1_gender_probs_by_condition{suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# fig2_token_association_table
# Qwen simple (female5_patch_male) + OLMo at layer 18, condition-token mean
# OLMo sarcoidosis → NaN → render as "n/a"
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2_token_table():
    print("fig2_token_association_table ...")
    qwen_csv = os.path.join(REPO,
        "activation_patching/simple_patching/female5_patch_male/"
        "condition_token_analysis/aggregate_by_cohort_layer_condition_token_summary.csv")
    olmo_csv = os.path.join(REPO,
        "activation_patching/simple_patching/olmo31_rewrite_only/"
        "condition_token_analysis/aggregate_by_cohort_layer_condition_token_summary.csv")

    for p in (qwen_csv, olmo_csv):
        if not os.path.exists(p):
            skip("fig2_token_association_table", f"missing {p}")
            return

    qwen = pd.read_csv(qwen_csv)
    olmo = pd.read_csv(olmo_csv)

    q18 = qwen[(qwen["layer"] == 18) & (qwen["score_key"] == "rewrite_scores")].copy()
    o18 = olmo[(olmo["layer"] == 18) & (olmo["score_key"] == "rewrite_scores")].copy()

    COND_ORDER = ["asthma", "depression", "multiple_sclerosis",
                  "rheumatoid_arthritis", "sarcoidosis"]
    PRETTY = {
        "asthma": "Asthma",
        "depression": "Depression",
        "multiple_sclerosis": "Multiple Sclerosis",
        "rheumatoid_arthritis": "Rheumatoid Arthritis",
        "sarcoidosis": "Sarcoidosis",
    }

    rows = []
    for c in COND_ORDER:
        q_row = q18[q18["cohort"] == c]
        o_row = o18[o18["cohort"] == c]
        q_val = q_row["condition_token_mean"].values[0] if len(q_row) else np.nan
        o_val = o_row["condition_token_mean"].values[0] if len(o_row) else np.nan
        rows.append({
            "condition": PRETTY.get(c, c),
            "Qwen2.5-7B": q_val,
            "OLMo-7B": o_val,
        })
    df = pd.DataFrame(rows)

    # Format cell values: NaN → "n/a", else 2 decimal places
    def fmt(v):
        return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    col_labels = ["Condition", "Qwen2.5-7B", "OLMo-7B"]
    cell_data  = [[r["condition"], fmt(r["Qwen2.5-7B"]), fmt(r["OLMo-7B"])]
                  for _, r in df.iterrows()]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 1.8)

    # header row styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(DARK)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # alternate row shading
    for i in range(1, len(cell_data) + 1):
        fc = "#f5e8e8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)
            # highlight n/a in OLMo column
            if cell_data[i - 1][2] == "n/a":
                tbl[i, 2].set_text_props(color="gray", style="italic")

    ax.set_title(
        "Condition-token rewrite score at layer 18\n"
        "(mean across units; OLMo sarcoidosis = tokenization-span failure)",
        fontsize=10, pad=12)
    plt.tight_layout()
    save("fig2_token_association_table")

    # Confirm sarcoidosis OLMo NaN
    o_sarc = o18[o18["cohort"] == "sarcoidosis"]["condition_token_mean"].values
    is_nan = len(o_sarc) == 0 or np.isnan(o_sarc[0])
    print(f"  OLMo sarcoidosis L18 is NaN: {is_nan}  "
          f"→ rendered as {'n/a' if is_nan else o_sarc[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# fig2c_toplayers_all_conditions  (Step 3a: exclude layers 0 AND 2)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2c_toplayers():
    print("fig2c_toplayers_all_conditions ...")
    path = os.path.join(REPO,
        "activation_patching/simple_patching/female5_patch_male/aggregate_per_layer.json")
    if not os.path.exists(path):
        skip("fig2c_toplayers_all_conditions", f"missing {path}")
        return

    with open(path) as f:
        a = json.load(f)

    n_layers, per_layer, counts = 28, np.zeros(28), np.zeros(28)
    for unit in a["raw_units"]:
        for l, v in enumerate(unit["score_stats"]["rewrite_scores"]["mean"]):
            per_layer[l] += v
            counts[l]    += 1
    per_layer /= np.maximum(counts, 1)

    EXCLUDE = {0, 2}
    layers  = [l for l in range(n_layers) if l not in EXCLUDE]
    scores  = [per_layer[l] for l in layers]

    top_idx    = np.argsort(scores)[::-1][:15]
    top_layers = [layers[i] for i in top_idx]
    top_scores = [scores[i]  for i in top_idx]

    print(f"  → Top layer after excluding 0,2: L{top_layers[0]}  "
          f"(score={top_scores[0]:.4f})")
    if top_layers[0] != 18:
        print(f"  WARNING: expected L18, got L{top_layers[0]}")

    colors = [DARK if l == 18 else MID for l in top_layers]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(top_layers)), top_scores, color=colors)
    ax.set_xticks(range(len(top_layers)))
    ax.set_xticklabels([f"L{l}" for l in top_layers], fontsize=9)
    ax.set_ylabel("Mean rewrite score (all units, all conditions)", fontsize=10)
    ax.set_title(
        "Top layers by mean rewrite score\n"
        "(layers 0 and 2 excluded — early-layer template artifacts)", fontsize=11)
    ax.set_xlabel("Layer (layers 0, 2 excluded)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save("fig2c_toplayers_all_conditions")


# ─────────────────────────────────────────────────────────────────────────────
# fig11_residual_plateau_qwen
# ─────────────────────────────────────────────────────────────────────────────

def make_fig11_residual_plateau():
    print("fig11_residual_plateau_qwen ...")
    path = os.path.join(REPO,
        "raw_uploads/simple_prompt_residual/aggregate_per_layer.csv")
    if not os.path.exists(path):
        skip("fig11_residual_plateau_qwen", f"missing {path}")
        return

    df = pd.read_csv(path)
    layers = df["layer"].values
    mean   = df["rewrite_scores_mean"].values
    median = df["rewrite_scores_median"].values
    topk   = df["rewrite_scores_topk_mean"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, mean,   color=DARK,  linewidth=2.0, label="Mean")
    ax.plot(layers, median, color=MID,   linewidth=1.5, linestyle="--", label="Median")
    ax.plot(layers, topk,   color=LIGHT, linewidth=1.5, linestyle=":",  label="Top-k mean")
    ax.axvline(18, color=DARK, linewidth=0.9, linestyle="--", alpha=0.6)
    ax.annotate("L18", xy=(18, mean[18]), xytext=(6, 4),
                textcoords="offset points", fontsize=9, color=DARK)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean rewrite score\n(all units, all conditions)", fontsize=11)
    ax.set_title(
        "Residual-stream rewrite score plateau — Qwen2.5-7B\n"
        "(simple prompts, female→male patching)", fontsize=11)
    ax.set_xticks(range(0, len(layers), 2))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save("fig11_residual_plateau_qwen")


# ─────────────────────────────────────────────────────────────────────────────
# simple_vs_cot_l18_comparison
# Data: sam/cot-behavioral-n35 simple_vs_cot_l18_comparison.csv
# ─────────────────────────────────────────────────────────────────────────────

def make_simple_vs_cot():
    print("simple_vs_cot_l18_comparison ...")
    csv_path = os.path.join(COT_DIR, "simple_vs_cot_l18_comparison.csv")
    if not os.path.exists(csv_path):
        skip("simple_vs_cot_l18_comparison", f"missing {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Sanity check
    all_row = df[df["condition"] == "ALL"]
    if len(all_row):
        r = all_row.iloc[0]
        print(f"  Sanity ALL: simple={r['simple_L18_condition_token']:.4f} "
              f"(exp 0.3975), cot_first={r['cot_L18_first_condition']:.4f} "
              f"(exp 0.4434), cot_last={r['cot_L18_last_condition']:.4f} (exp 0.0494)")
        asthma = df[df["condition"] == "asthma"].iloc[0]
        print(f"  Sanity asthma: simple={asthma['simple_L18_condition_token']:.3f} "
              f"(exp 0.833), cot_first={asthma['cot_L18_first_condition']:.3f} (exp 0.936)")

    # Plot per-condition grouped bar chart (match Om's layout)
    conds = df[df["condition"] != "ALL"]["condition"].tolist()
    x     = np.arange(len(conds))
    w     = 0.25

    pretty = {
        "asthma": "Asthma", "depression": "Depression",
        "multiple_sclerosis": "MS", "rheumatoid_arthritis": "RA",
        "sarcoidosis": "Sarcoidosis",
    }
    xlabels = [pretty.get(c, c) for c in conds]

    cond_df = df[df["condition"] != "ALL"].reset_index(drop=True)
    s_vals  = cond_df["simple_L18_condition_token"].values
    c1_vals = cond_df["cot_L18_first_condition"].values
    cl_vals = cond_df["cot_L18_last_condition"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w,   s_vals,  w, label="Simple L18",           color=DARK)
    ax.bar(x,       c1_vals, w, label="CoT L18 first-cond",   color=MID)
    ax.bar(x + w,   cl_vals, w, label="CoT L18 last-cond",    color=LIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Condition-token rewrite score (layer 18)", fontsize=10)
    ax.set_title(
        "Simple vs CoT patching at layer 18 — per condition\n"
        "(Qwen2.5-7B, female→male patching)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save("simple_vs_cot_l18_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# fig3_mlp_heatmaps_combined  /  fig5_mlp_heatmaps_combined_asthma_ra
# Step 3b: shared TwoSlopeNorm, 99th-pct of |score| from layer 1+
# ─────────────────────────────────────────────────────────────────────────────

def _load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["rewrite_scores"], d["token_labels"]  # (28, n_tokens), list


def _shared_vmax(pkls):
    vals = []
    for p in pkls:
        mat, _ = _load_pkl(p)
        vals.append(np.abs(mat[1:]).ravel())
    return float(np.percentile(np.concatenate(vals), 99))


def _top_tokens(mat, n=18):
    return np.argsort(np.abs(mat[1:]).max(axis=0))[::-1][:n]


def _heatmap_panel(ax, mat, tokens, top_idx, norm, title):
    sub = mat[:, top_idx].T
    im  = ax.imshow(sub, aspect="auto", cmap=DIVERGING, norm=norm, origin="upper")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([tokens[i].split("_")[0] for i in top_idx], fontsize=7)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_xticks(range(0, mat.shape[0], 4))
    ax.set_xticklabels(range(0, mat.shape[0], 4), fontsize=7)
    ax.set_title(title, fontsize=9)
    return im


def make_mlp_heatmaps():
    arts = os.path.join(REPO,
        "activation_patching/simple_patching/female5_patch_male/artifacts")

    # fig3: all 5 conditions
    print("fig3_mlp_heatmaps_combined ...")
    COHORTS5 = ["asthma", "depression", "multiple_sclerosis",
                "rheumatoid_arthritis", "sarcoidosis"]
    pkls5 = [os.path.join(arts, f"{c}_prompt1.pkl") for c in COHORTS5]
    if any(not os.path.exists(p) for p in pkls5):
        skip("fig3_mlp_heatmaps_combined",
             f"missing pkl(s): {[p for p in pkls5 if not os.path.exists(p)]}")
    else:
        vmax = _shared_vmax(pkls5)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        im = None
        for ax, coh, pkl in zip(axes, COHORTS5, pkls5):
            mat, toks = _load_pkl(pkl)
            top_idx   = _top_tokens(mat)
            im = _heatmap_panel(ax, mat, toks, top_idx, norm,
                                coh.replace("_", " ").title())
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Rewrite score")
        fig.suptitle(
            "MLP rewrite scores by layer and token — all conditions\n"
            "(shared scale; layer 0 excluded from scale computation)", fontsize=11)
        plt.subplots_adjust(right=0.91, wspace=0.4)
        save("fig3_mlp_heatmaps_combined")

    # fig5: asthma + RA
    print("fig5_mlp_heatmaps_combined_asthma_ra ...")
    COHORTS2 = ["asthma", "rheumatoid_arthritis"]
    pkls2 = [os.path.join(arts, f"{c}_prompt1.pkl") for c in COHORTS2]
    if any(not os.path.exists(p) for p in pkls2):
        skip("fig5_mlp_heatmaps_combined_asthma_ra",
             f"missing pkl(s): {[p for p in pkls2 if not os.path.exists(p)]}")
        return
    vmax2 = _shared_vmax(pkls2)
    norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im = None
    for ax, coh, pkl in zip(axes, COHORTS2, pkls2):
        mat, toks = _load_pkl(pkl)
        top_idx   = _top_tokens(mat)
        im = _heatmap_panel(ax, mat, toks, top_idx, norm2,
                            coh.replace("_", " ").title())
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Rewrite score")
    fig.suptitle(
        "MLP rewrite scores: Asthma vs Rheumatoid Arthritis\n"
        "(shared colour scale — shows true relative magnitude)", fontsize=11)
    plt.subplots_adjust(right=0.92, wspace=0.35)
    save("fig5_mlp_heatmaps_combined_asthma_ra")


# ─────────────────────────────────────────────────────────────────────────────
# fig1_heatmaps_A_C  — SAE layer×token heatmap, Prompt A and C panels
# Reconstructed from sweep_results.parquet across all runs per family
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1_heatmaps_A_C():
    print("fig1_heatmaps_A_C ...")
    if not os.path.isdir(SAE_RUNS):
        skip("fig1_heatmaps_A_C", f"SAE runs dir missing: {SAE_RUNS}")
        return

    import pyarrow.parquet as pq_mod

    # Map family name from metadata
    with open(os.path.join(SAE_DIR, "metadata.json")) as f:
        meta = json.load(f)
    # metadata paths use Windows backslashes; split on both separators
    run_c_names = {r.replace("\\", "/").split("/")[-1] for r in meta["runs_c"]}
    run_a_names = {r.replace("\\", "/").split("/")[-1] for r in meta["runs_a"]}

    # Load and concatenate all runs per family
    frames = {"A": [], "C": []}
    for run_dir in sorted(glob.glob(os.path.join(SAE_RUNS, "*"))):
        name   = os.path.basename(run_dir)
        pq_path = os.path.join(run_dir, "artifacts", "sweep_results.parquet")
        if not os.path.exists(pq_path):
            continue
        df = pd.read_parquet(pq_path)
        if name in run_a_names:
            frames["A"].append(df)
        elif name in run_c_names:
            frames["C"].append(df)

    if not frames["A"] or not frames["C"]:
        skip("fig1_heatmaps_A_C", "No parquet files loaded for one or both families")
        return

    TOP_TOKENS = meta.get("top_tokens", 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    FAMILY_LABELS = {"A": "Prompt A mean", "C": "Prompt C mean"}

    for ax, family in zip(axes, ["A", "C"]):
        df = pd.concat(frames[family], ignore_index=True)

        # Mean norm_effect per (layer, token_text)
        grouped = (df.groupby(["layer", "token_text"])["norm_effect"]
                     .mean()
                     .reset_index())

        # Top tokens by peak |mean norm_effect| across layers
        token_peak = grouped.groupby("token_text")["norm_effect"].apply(
            lambda x: x.abs().max())
        top_tokens = token_peak.nlargest(TOP_TOKENS).index.tolist()

        layers = sorted(grouped["layer"].unique())
        mat = np.zeros((len(layers), len(top_tokens)))
        for i, l in enumerate(layers):
            sub = grouped[grouped["layer"] == l].set_index("token_text")["norm_effect"]
            for j, tok in enumerate(top_tokens):
                mat[i, j] = sub.get(tok, 0.0)

        vmax = np.abs(mat).max()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im   = ax.imshow(mat.T, aspect="auto", cmap=DIVERGING, norm=norm, origin="upper")

        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens, fontsize=7)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers], rotation=90, fontsize=7)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("Token identity", fontsize=9)
        ax.set_title(FAMILY_LABELS[family], fontsize=11)
        fig.colorbar(im, ax=ax, label="Mean norm effect", shrink=0.8)

    fig.suptitle(
        "SAE latent activation by layer and token — Prompt A vs C\n"
        "(positive gate; top 40 tokens by peak |mean norm effect|)", fontsize=11)
    plt.tight_layout()
    save("fig1_heatmaps_A_C")


# ─────────────────────────────────────────────────────────────────────────────
# fig2_top10_latents_bootstrap_ci
# Step 3c resolution: top_latents=10, L20:F26533 excluded (n=10 for A, C=NaN)
# → 9 bars plotted; title updated to state "top 10 selected, 9 plotted"
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2_top10():
    print("fig2_top10_latents_bootstrap_ci ...")
    stats_path  = os.path.join(SAE_DIR, "fig2_latent_bootstrap_stats.csv")
    pooled_path = os.path.join(SAE_DIR, "fig2_pooled_top_latents.csv")
    if not os.path.exists(stats_path) or not os.path.exists(pooled_path):
        skip("fig2_top10_latents_bootstrap_ci",
             f"missing {stats_path} or {pooled_path}")
        return

    stats  = pd.read_csv(stats_path)
    pooled = pd.read_csv(pooled_path)

    # The 10 selected latents (from metadata top_latents=10)
    # L20:F26533 has A n=10 → excluded from plot (both filters n>20 and n>=25 exclude it)
    # Plotted: 9 latents (all rows where BOTH families have n_rows > 0 and are non-NaN,
    # excluding the L20:F26533 entry)

    # Build plot: one grouped bar per latent, Prompt A and C side by side
    # Sort by pooled mean descending (as in the original committed figure)
    latent_order = pooled.apply(
        lambda r: f"L{int(r['layer'])}:F{int(r['feature_idx'])}", axis=1).tolist()

    # Filter stats to rows we can plot (non-NaN mean_norm_effect)
    plottable = stats[stats["mean_norm_effect"].notna()].copy()
    plottable["latent_label"] = plottable.apply(
        lambda r: f"L{int(r['layer'])}:F{int(r['feature_idx'])}", axis=1)

    # Keep only latents in pooled list (top 10 selected), exclude those with n_rows <= 0 in C
    valid_latents = [l for l in latent_order
                     if l in plottable["latent_label"].values]
    # Exclude L20:F26533 (C has NaN/0)
    valid_latents = [l for l in valid_latents
                     if l != "L20:F26533"]

    n_plotted = len(valid_latents)
    print(f"  Top 10 selected, {n_plotted} plotted (L20:F26533 excluded — A n=10, C n=0)")

    x   = np.arange(n_plotted)
    w   = 0.35
    a_vals, a_lo, a_hi = [], [], []
    c_vals, c_lo, c_hi = [], [], []

    for lab in valid_latents:
        a = plottable[(plottable["latent_label"] == lab) & (plottable["family"] == "A")]
        c = plottable[(plottable["latent_label"] == lab) & (plottable["family"] == "C")]
        a_vals.append(a["mean_norm_effect"].values[0] if len(a) else 0)
        a_lo.append(a["ci_low"].values[0]  if len(a) else 0)
        a_hi.append(a["ci_high"].values[0] if len(a) else 0)
        c_vals.append(c["mean_norm_effect"].values[0] if len(c) else 0)
        c_lo.append(c["ci_low"].values[0]  if len(c) else 0)
        c_hi.append(c["ci_high"].values[0] if len(c) else 0)

    a_vals = np.array(a_vals); a_lo = np.array(a_lo); a_hi = np.array(a_hi)
    c_vals = np.array(c_vals); c_lo = np.array(c_lo); c_hi = np.array(c_hi)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, a_vals, w, label="Prompt A", color=DARK,
           yerr=[a_vals - a_lo, a_hi - a_vals], capsize=3, error_kw=dict(linewidth=0.8))
    ax.bar(x + w/2, c_vals, w, label="Prompt C", color=MID,
           yerr=[c_vals - c_lo, c_hi - c_vals], capsize=3, error_kw=dict(linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(valid_latents, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean norm effect (95% bootstrap CI)", fontsize=10)
    ax.set_title(
        f"Top 10 SAE latents by pooled mean norm effect\n"
        f"({n_plotted} plotted — L20:F26533 excluded, Prompt C n=0)\n"
        f"2000 bootstrap iterations, positive gate", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save("fig2_top10_latents_bootstrap_ci")


# ─────────────────────────────────────────────────────────────────────────────
# fig2_top10_latents_bootstrap_ci_simple — no committed stats CSV → SKIP
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2_top10_simple():
    skip("fig2_top10_latents_bootstrap_ci_simple",
         "No bootstrap stats CSV committed for simple-prompt SAE runs. "
         "7-bar vs 10-bar discrepancy cannot be resolved from code. "
         "Leave existing committed PNG unchanged.")


# ─────────────────────────────────────────────────────────────────────────────
# fig3_real_vs_control_gap  /  fig3_ci_real_vs_controls
# Data: fig3_control_bootstrap_stats.csv (activation_patching_simple branch)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig3_control_figures():
    print("fig3_real_vs_control_gap + fig3_ci_real_vs_controls ...")
    stats_path = os.path.join(SAE_DIR, "fig3_control_bootstrap_stats.csv")
    if not os.path.exists(stats_path):
        skip("fig3_real_vs_control_gap", f"missing {stats_path}")
        skip("fig3_ci_real_vs_controls", f"missing {stats_path}")
        return

    df = pd.read_csv(stats_path)
    SERIES_LABELS = {
        "real": "Real ablations",
        "condition_semantic": "Condition semantic controls",
        "random_magnitude_matched": "Random magnitude-matched",
    }
    FAMILY_COLORS = {"A": DARK, "C": MID}

    # fig3_real_vs_control_gap: gap (real minus each control) per family
    gap_rows = df[df["series"] != "real"].copy()
    gap_rows["series_label"] = gap_rows["series"].map(SERIES_LABELS)

    fig, ax = plt.subplots(figsize=(7, 4))
    families = ["A", "C"]
    n_series  = gap_rows["series"].nunique()
    x         = np.arange(n_series)
    w         = 0.35

    for i, fam in enumerate(families):
        sub = gap_rows[gap_rows["family"] == fam]
        gaps = sub["gap_real_minus_series"].values
        labels = sub["series_label"].values
        ax.bar(x + (i - 0.5) * w, gaps, w,
               label=f"Prompt {fam}", color=FAMILY_COLORS[fam])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Gap: real − control (mean norm effect)", fontsize=10)
    ax.set_title(
        "Real ablation advantage over controls\n"
        "(simple prompts, Prompt A and C families)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save("fig3_real_vs_control_gap")

    # fig3_ci_real_vs_controls: error-bar plot of mean ± CI for all series
    print("fig3_ci_real_vs_controls ...")
    series_order = ["real", "condition_semantic", "random_magnitude_matched"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, fam in zip(axes, families):
        sub = df[df["family"] == fam].copy()
        sub["series_label"] = sub["series"].map(SERIES_LABELS)
        ys      = [sub[sub["series"] == s]["mean_norm_effect"].values[0]
                   for s in series_order]
        y_lo    = [sub[sub["series"] == s]["ci_low"].values[0]
                   for s in series_order]
        y_hi    = [sub[sub["series"] == s]["ci_high"].values[0]
                   for s in series_order]
        xlabels = [SERIES_LABELS[s] for s in series_order]
        xs      = range(len(series_order))
        colors  = [DARK, MID, LIGHT]

        for xi, (y, lo, hi, col) in enumerate(zip(ys, y_lo, y_hi, colors)):
            ax.errorbar(xi, y, yerr=[[y - lo], [hi - y]],
                        fmt="o", color=col, capsize=5, markersize=7, linewidth=1.5)

        ax.set_xticks(list(xs))
        ax.set_xticklabels(xlabels, rotation=15, ha="right", fontsize=8)
        ax.set_title(f"Prompt {fam}", fontsize=11)
        ax.set_ylabel("Mean norm effect (95% CI)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Real ablations vs controls — mean norm effect with 95% bootstrap CI\n"
        "(simple prompts, positive SAE gate)", fontsize=11)
    plt.tight_layout()
    save("fig3_ci_real_vs_controls")


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks():
    print("\n=== SANITY CHECKS ===")

    cot_csv = os.path.join(COT_DIR, "simple_vs_cot_l18_comparison.csv")
    if os.path.exists(cot_csv):
        df = pd.read_csv(cot_csv)
        all_row = df[df["condition"] == "ALL"].iloc[0]
        asthma  = df[df["condition"] == "asthma"].iloc[0]
        print(f"\nsimple_vs_cot_l18_comparison.csv ALL row:")
        print(f"  simple_L18_condition_token : {all_row['simple_L18_condition_token']:.4f}  (expected 0.3975)")
        print(f"  cot_L18_first_condition    : {all_row['cot_L18_first_condition']:.4f}  (expected 0.4434)")
        print(f"  cot_L18_last_condition     : {all_row['cot_L18_last_condition']:.4f}  (expected 0.0494)")
        print(f"  asthma simple              : {asthma['simple_L18_condition_token']:.3f}   (expected 0.833)")
        print(f"  asthma cot_first           : {asthma['cot_L18_first_condition']:.3f}   (expected 0.936)")

        ok = (
            abs(all_row["simple_L18_condition_token"] - 0.3975) < 0.001 and
            abs(all_row["cot_L18_first_condition"]    - 0.4434) < 0.001 and
            abs(all_row["cot_L18_last_condition"]     - 0.0494) < 0.001 and
            abs(asthma["simple_L18_condition_token"]  - 0.833)  < 0.002 and
            abs(asthma["cot_L18_first_condition"]     - 0.936)  < 0.002
        )
        print(f"  {'✓ ALL values match expected' if ok else '✗ MISMATCH — check above'}")

    olmo_csv = os.path.join(REPO,
        "activation_patching/simple_patching/olmo31_rewrite_only/"
        "condition_token_analysis/aggregate_by_cohort_layer_condition_token_summary.csv")
    if os.path.exists(olmo_csv):
        df = pd.read_csv(olmo_csv)
        row = df[(df["layer"] == 18) & (df["score_key"] == "rewrite_scores") &
                 (df["cohort"] == "sarcoidosis")]
        val = row["condition_token_mean"].values
        is_nan = len(val) == 0 or np.isnan(val[0])
        print(f"\nOLMo sarcoidosis L18: is NaN = {is_nan}  (expected True)")
        print(f"  {'✓ will render as n/a' if is_nan else '✗ WARNING: not NaN, check source'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generating RTAR paper figures ===\n")

    make_fig1_gender_probs()
    make_fig2_token_table()
    make_fig2c_toplayers()
    make_fig11_residual_plateau()
    make_simple_vs_cot()
    make_mlp_heatmaps()
    make_fig1_heatmaps_A_C()
    make_fig2_top10()
    make_fig2_top10_simple()
    make_fig3_control_figures()

    run_sanity_checks()

    print("\n=== SKIPPED FIGURES ===")
    for name, reason in SKIPPED:
        print(f"\n  {name}:\n    {reason}")

    out_files = (glob.glob(os.path.join(OUT, "*.png")) +
                 glob.glob(os.path.join(OUT, "*.pdf")))
    print(f"\n=== DONE — {len(out_files)} files in {OUT} ===")
