"""
Psycholinguistic Feature Analysis for Adversarial Deception Detection
====================================================================

Focuses on the 4 features explicitly targeted in the attack prompt:
  1. Perceptual process language
  2. Affective intensity
  3. Self-referential pronouns
  4. Balanced certainty expressions

This is a LIWC-inspired proxy analysis (not official LIWC software).
"""

import os
import re
import ast
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu

# -------------------------------------------------------------------
# 1) LIWC-inspired lexicons for the 4 targeted prompt features
# -------------------------------------------------------------------

TOKEN_RE = re.compile(r"\b[\w'-]+\b")

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves"
}

# Perceptual / sensory words (Reality Monitoring inspired)
PERCEPTUAL_WORDS = {
    # visual
    "see", "saw", "seen", "look", "looked", "looks", "view", "viewed",
    "watch", "watched", "bright", "dark", "shiny", "color", "window",
    # auditory
    "hear", "heard", "listen", "listened", "sound", "sounds", "quiet",
    "loud", "noise", "noisy",
    # touch / feeling / bodily sensation
    "feel", "felt", "feeling", "touch", "touched", "soft", "hard",
    "warm", "cold", "smooth", "rough", "comfortable", "uncomfortable",
    "clean", "dirty",
    # smell / taste
    "smell", "smelled", "scent", "odor", "taste", "tasted"
}

# Affect words: keep total + positive + negative separately
POS_AFFECT_WORDS = {
    "good", "great", "amazing", "excellent", "nice", "pleasant", "enjoyed",
    "enjoy", "happy", "love", "loved", "lovely", "wonderful", "fantastic",
    "perfect", "beautiful", "clean", "comfortable", "helpful", "friendly",
    "satisfied", "recommend"
}

NEG_AFFECT_WORDS = {
    "bad", "awful", "terrible", "horrible", "poor", "dirty", "uncomfortable",
    "disappointed", "disappointing", "annoying", "angry", "upset", "hate",
    "hated", "worst", "smelly", "noisy", "rude", "problem", "problems",
    "frustrating", "frustrated"
}

# Certainty vs. tentative / hedge words
CERTAINTY_WORDS = {
    "always", "never", "definitely", "certainly", "clearly", "obviously",
    "surely", "undoubtedly", "absolutely", "guaranteed", "everyone",
    "nobody", "must"
}

TENTATIVE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "seems", "seemed", "appear",
    "appears", "appeared", "guess", "guessed", "kind", "sort", "somewhat",
    "likely", "might", "could", "may"
}

FEATURES = [
    "Perceptual_Rate",
    "Affect_Total_Rate",
    "Positive_Affect_Rate",
    "Negative_Affect_Rate",
    "Self_Ref_Rate",
    "Certainty_Rate",
    "Tentative_Rate",
    "Certainty_Balance_Index",
]

PRIMARY_FEATURES = [
    "Perceptual_Rate",
    "Affect_Total_Rate",
    "Self_Ref_Rate",
    "Certainty_Balance_Index",
]

PRIMARY_LABELS = {
    "Perceptual_Rate": "Perceptual language",
    "Affect_Total_Rate": "Affective intensity",
    "Self_Ref_Rate": "Self-reference",
    "Certainty_Balance_Index": "Certainty balance"
}

# -------------------------------------------------------------------
# 2) Tokenization + feature extraction
# -------------------------------------------------------------------

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())

def safe_rate(count, total):
    return count / total if total > 0 else 0.0

def extract_features(text: str) -> dict:
    """
    Returns LIWC-inspired proxy feature scores as token-normalized rates.
    """
    if not isinstance(text, str) or not text.strip():
        return {f: np.nan for f in FEATURES}

    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return {f: np.nan for f in FEATURES}

    perceptual_count = sum(t in PERCEPTUAL_WORDS for t in tokens)

    pos_aff_count = sum(t in POS_AFFECT_WORDS for t in tokens)
    neg_aff_count = sum(t in NEG_AFFECT_WORDS for t in tokens)
    affect_total_count = pos_aff_count + neg_aff_count

    self_ref_count = sum(t in FIRST_PERSON_PRONOUNS for t in tokens)

    certainty_count = sum(t in CERTAINTY_WORDS for t in tokens)
    tentative_count = sum(t in TENTATIVE_WORDS for t in tokens)

    perceptual_rate = safe_rate(perceptual_count, n)
    pos_aff_rate = safe_rate(pos_aff_count, n)
    neg_aff_rate = safe_rate(neg_aff_count, n)
    affect_total_rate = safe_rate(affect_total_count, n)
    self_ref_rate = safe_rate(self_ref_count, n)
    certainty_rate = safe_rate(certainty_count, n)
    tentative_rate = safe_rate(tentative_count, n)

    # "Balanced certainty" = avoid being heavily one-sided
    certainty_balance_index = abs(certainty_rate - tentative_rate)

    return {
        "Perceptual_Rate": perceptual_rate,
        "Affect_Total_Rate": affect_total_rate,
        "Positive_Affect_Rate": pos_aff_rate,
        "Negative_Affect_Rate": neg_aff_rate,
        "Self_Ref_Rate": self_ref_rate,
        "Certainty_Rate": certainty_rate,
        "Tentative_Rate": tentative_rate,
        "Certainty_Balance_Index": certainty_balance_index,
    }


# -------------------------------------------------------------------
# 3) History parsing
# -------------------------------------------------------------------

def get_last_accepted_candidate_from_history(history_str: str):
    """
    Prefer the last accepted candidate.
    If none exists, fall back to the last non-empty candidate.
    """
    try:
        history = ast.literal_eval(history_str)
    except Exception:
        return None

    if not isinstance(history, list) or len(history) == 0:
        return None

    # First pass: last accepted candidate
    for step in reversed(history):
        text = step.get("text", "")
        accepted = step.get("accepted", True)  # default True if field missing
        if isinstance(text, str) and text.strip() and accepted:
            return text.strip()

    # Fallback: last non-empty candidate
    for step in reversed(history):
        text = step.get("text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

    return None


# -------------------------------------------------------------------
# 4) Load attack results
# -------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    dfs = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv"):
            path = os.path.join(results_dir, fname)
            try:
                dfs.append(pd.read_csv(path))
            except Exception as e:
                print(f"Warning: could not read {fname}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(dfs)} files.")
    return df


# -------------------------------------------------------------------
# 5) Truthful baseline (important for 'natural' / 'balanced')
# -------------------------------------------------------------------

def load_truthful_reference(opspam_csv: str) -> pd.DataFrame:
    """
    Loads truthful hotel reviews from OpSpam.
    Expects columns:
      - text
      - deceptive with values {"truthful","deceptive"}
    """
    ref_df = pd.read_csv(opspam_csv)
    ref_df = ref_df[["text", "deceptive"]].dropna()
    truth_df = ref_df[ref_df["deceptive"].str.lower() == "truthful"].copy()

    truth_features = truth_df["text"].apply(extract_features).apply(pd.Series)
    truth_features["text"] = truth_df["text"].values
    return truth_features

def build_truthful_reference_stats(truth_features: pd.DataFrame) -> dict:
    """
    Store truthful medians per feature so we can compute:
      distance_to_truthful = abs(feature_value - truthful_median)
    """
    ref_stats = {}
    for feat in FEATURES:
        ref_stats[feat] = float(np.nanmedian(truth_features[feat].values))
    return ref_stats


# -------------------------------------------------------------------
# 6) Build comparison dataframe
# -------------------------------------------------------------------

def add_distance_to_truthful(feature_dict: dict, truthful_medians: dict) -> dict:
    out = feature_dict.copy()
    for feat in FEATURES:
        out[f"{feat}_DistToTruth"] = abs(out[feat] - truthful_medians[feat])
    return out

def build_comparison_df(df: pd.DataFrame, truthful_medians: dict) -> pd.DataFrame:
    """
    Creates rows for:
      - original deceptive review
      - successful paraphrase
      - failed paraphrase (last accepted candidate)
    And adds distance-to-truthful metrics for each feature.
    """
    records = []
    seen_originals = set()

    for _, row in df.iterrows():
        sid = row.get("sample_id")
        original_text = row.get("original_text", "")
        adversarial_text = row.get("adversarial_text", "")
        success = bool(row.get("success", False))
        history_str = str(row.get("history", "[]"))

        # Original (only once per sample_id)
        if sid not in seen_originals:
            feats = extract_features(original_text)
            feats = add_distance_to_truthful(feats, truthful_medians)
            records.append({
                "sample_id": sid,
                "attacker": "all",
                "victim": "all",
                "group": "original",
                **feats
            })
            seen_originals.add(sid)

        # Successful attack
        if success and isinstance(adversarial_text, str) and adversarial_text.strip():
            feats = extract_features(adversarial_text)
            feats = add_distance_to_truthful(feats, truthful_medians)
            records.append({
                "sample_id": sid,
                "attacker": row.get("attacker"),
                "victim": row.get("victim"),
                "group": "successful",
                **feats
            })

        # Failed attack
        elif not success:
            last_cand = get_last_accepted_candidate_from_history(history_str)
            if last_cand:
                feats = extract_features(last_cand)
                feats = add_distance_to_truthful(feats, truthful_medians)
                records.append({
                    "sample_id": sid,
                    "attacker": row.get("attacker"),
                    "victim": row.get("victim"),
                    "group": "failed",
                    **feats
                })

    result = pd.DataFrame(records)
    print(f"Built comparison DataFrame: {len(result)} rows")
    print(result["group"].value_counts().to_string())
    return result


# -------------------------------------------------------------------
# 7) Statistical tests
# -------------------------------------------------------------------

def run_statistics(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired:   original vs successful   (Wilcoxon)
    Unpaired: successful vs failed     (Mann-Whitney)

    We test BOTH:
      - raw feature values
      - distance-to-truthful values
    """
    test_features = FEATURES + [f"{feat}_DistToTruth" for feat in FEATURES]
    n_tests = len(test_features)

    rows = []

    orig = comp_df[comp_df["group"] == "original"].set_index("sample_id")

    succ = (
        comp_df[comp_df["group"] == "successful"]
        .groupby("sample_id")
        .mean(numeric_only=True)
    )

    fail = (
        comp_df[comp_df["group"] == "failed"]
        .groupby("sample_id")
        .mean(numeric_only=True)
    )

    for feat in test_features:
        paired_df = pd.DataFrame({
            "orig": orig[feat],
            "succ": succ[feat]
        }).dropna()

        o_arr = paired_df["orig"].values
        s_arr = paired_df["succ"].values

        if len(paired_df) >= 10:
            diff = s_arr - o_arr
            if np.all(diff == 0):
                w_stat, w_p = np.nan, 1.0
            else:
                w_stat, w_p = wilcoxon(o_arr, s_arr, alternative="two-sided")
        else:
            w_stat, w_p = np.nan, np.nan

        s_vals = succ[feat].dropna().values
        f_vals = fail[feat].dropna().values

        if len(s_vals) >= 5 and len(f_vals) >= 5:
            mw_stat, mw_p = mannwhitneyu(s_vals, f_vals, alternative="two-sided")
        else:
            mw_stat, mw_p = np.nan, np.nan

        rows.append({
            "Feature": feat,
            "n_paired": len(paired_df),
            "Median_Original": np.nanmedian(o_arr) if len(o_arr) else np.nan,
            "Median_Successful": np.nanmedian(s_arr) if len(s_arr) else np.nan,
            "Median_Failed": np.nanmedian(f_vals) if len(f_vals) else np.nan,
            "Delta_Orig_to_Succ": (
                np.nanmedian(s_arr) - np.nanmedian(o_arr)
                if len(o_arr) and len(s_arr) else np.nan
            ),
            "Wilcoxon_p_raw": w_p,
            "MW_p_raw": mw_p,
        })

    stats_df = pd.DataFrame(rows)

    stats_df["Wilcoxon_p_bonf"] = (stats_df["Wilcoxon_p_raw"] * n_tests).clip(upper=1.0)
    stats_df["MW_p_bonf"] = (stats_df["MW_p_raw"] * n_tests).clip(upper=1.0)

    stats_df["Wilcoxon_sig"] = stats_df["Wilcoxon_p_bonf"] < 0.05
    stats_df["MW_sig"] = stats_df["MW_p_bonf"] < 0.05

    return stats_df


# -------------------------------------------------------------------
# 8) Plot helpers
# -------------------------------------------------------------------

def bootstrap_ci(values, func=np.nanmedian, n_boot=2000, ci=95, random_state=42):
    """
    Bootstrap CI for a one-sample statistic.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(func(sample))
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return lower, upper

def bootstrap_ci_paired_delta(orig_vals, succ_vals, func=np.nanmedian, n_boot=2000, ci=95, random_state=42):
    """
    Bootstrap CI for paired delta statistic on (succ - orig).
    """
    orig_vals = np.asarray(orig_vals, dtype=float)
    succ_vals = np.asarray(succ_vals, dtype=float)

    mask = ~(np.isnan(orig_vals) | np.isnan(succ_vals))
    orig_vals = orig_vals[mask]
    succ_vals = succ_vals[mask]

    if len(orig_vals) < 2:
        return np.nan, np.nan, np.nan

    deltas = succ_vals - orig_vals
    point = func(deltas)

    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(len(deltas)), size=len(deltas), replace=True)
        stats.append(func(deltas[idx]))

    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return point, lower, upper

def cles_success_better(success_vals, failed_vals):
    """
    Returns probability that a randomly drawn successful attack
    is MORE truth-like than a randomly drawn failed one.
    Here 'better' means SMALLER distance-to-truth.
    """
    s = np.asarray(success_vals, dtype=float)
    f = np.asarray(failed_vals, dtype=float)
    s = s[~np.isnan(s)]
    f = f[~np.isnan(f)]

    if len(s) == 0 or len(f) == 0:
        return np.nan

    wins = 0.0
    total = 0.0
    for sv in s:
        for fv in f:
            total += 1.0
            if sv < fv:
                wins += 1.0
            elif sv == fv:
                wins += 0.5
    return wins / total if total > 0 else np.nan

def bootstrap_ci_cles(success_vals, failed_vals, n_boot=2000, ci=95, random_state=42):
    s = np.asarray(success_vals, dtype=float)
    f = np.asarray(failed_vals, dtype=float)
    s = s[~np.isnan(s)]
    f = f[~np.isnan(f)]

    if len(s) < 2 or len(f) < 2:
        return np.nan, np.nan, np.nan

    point = cles_success_better(s, f)

    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_boot):
        s_boot = rng.choice(s, size=len(s), replace=True)
        f_boot = rng.choice(f, size=len(f), replace=True)
        stats.append(cles_success_better(s_boot, f_boot))

    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return point, lower, upper


# -------------------------------------------------------------------
# 9) Plots
# -------------------------------------------------------------------

def plot_trajectory_to_truthfulness(comp_df: pd.DataFrame, output_dir: str):
    """
    Grouped categorical bar chart:
      x-axis = primary features
      bars   = Original / Failed / Successful
      value  = median distance-to-truthful baseline

    Lower = more truth-like.
    """
    plot_rows = []

    for feat in PRIMARY_FEATURES:
        dist_feat = f"{feat}_DistToTruth"
        for group in ["original", "failed", "successful"]:
            vals = comp_df.loc[comp_df["group"] == group, dist_feat].dropna().values
            med = np.nanmedian(vals) if len(vals) > 0 else np.nan
            plot_rows.append({
                "Feature": PRIMARY_LABELS[feat],
                "Group": group.capitalize(),
                "MedianDist": med
            })

    plot_df = pd.DataFrame(plot_rows)

    group_order = ["Original", "Failed", "Successful"]
    palette = {
        "Original": "#4C78A8",
        "Failed": "#E45756",
        "Successful": "#54A24B"
    }

    plt.figure(figsize=(11, 6.5))
    ax = sns.barplot(
        data=plot_df,
        x="Feature",
        y="MedianDist",
        hue="Group",
        hue_order=group_order,
        palette=palette
    )

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=8)

    ax.set_title(
        "Median Distance to Truthful Baseline by Feature and Attack Outcome",
        fontsize=14,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median distance to truthful baseline\n(lower = more truth-like)", fontsize=11)
    ax.legend(title="", frameon=False)
    sns.despine()

    plt.tight_layout()
    path = os.path.join(output_dir, "trajectory_categorical_bar.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

def plot_effect_size_forest(comp_df: pd.DataFrame, output_dir: str):
    """
    Horizontal 100% stacked bar chart.

    Each row = one primary feature
    Each bar = 100%
    Segments = Successful / Original / Failed

    Segment widths are based on each group's median distance-to-truthful baseline
    for that feature, normalized so that each feature row sums to 100%.

    Interpretation:
      - Smaller Successful segment = Successful paraphrases are closer to truthful style.
      - Larger Original / Failed segment = those groups remain more deviant from truthful style.
    """

    feature_order = PRIMARY_FEATURES
    feature_labels = [PRIMARY_LABELS[f] for f in feature_order]

    group_colors = {
        "Successful": "#54A24B",  # green
        "Original": "#4C78A8",    # blue
        "Failed": "#E45756"       # red
    }

    def median_dist(group_name, feat):
        dist_feat = f"{feat}_DistToTruth"
        vals = comp_df.loc[comp_df["group"] == group_name, dist_feat].dropna().values
        return float(np.nanmedian(vals)) if len(vals) > 0 else np.nan

    plot_rows = []
    for feat in feature_order:
        med_success = median_dist("successful", feat)
        med_original = median_dist("original", feat)
        med_failed = median_dist("failed", feat)

        total = 0.0
        vals = {
            "Successful": 0.0 if np.isnan(med_success) else med_success,
            "Original": 0.0 if np.isnan(med_original) else med_original,
            "Failed": 0.0 if np.isnan(med_failed) else med_failed,
        }
        total = sum(vals.values())

        if total == 0:
            percentages = {k: 0.0 for k in vals}
        else:
            percentages = {k: (v / total) * 100 for k, v in vals.items()}

        plot_rows.append({
            "Feature": PRIMARY_LABELS[feat],
            "Successful_pct": percentages["Successful"],
            "Original_pct": percentages["Original"],
            "Failed_pct": percentages["Failed"],
            "Successful_raw": vals["Successful"],
            "Original_raw": vals["Original"],
            "Failed_raw": vals["Failed"],
        })

    plot_df = pd.DataFrame(plot_rows)

    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(13.5, 7))

    # Stacked segments
    left_success = np.zeros(len(plot_df))
    left_original = plot_df["Successful_pct"].values
    left_failed = plot_df["Successful_pct"].values + plot_df["Original_pct"].values

    ax.barh(
        y_pos,
        plot_df["Successful_pct"].values,
        color=group_colors["Successful"],
        edgecolor="white",
        height=0.6,
        label="Successful"
    )

    ax.barh(
        y_pos,
        plot_df["Original_pct"].values,
        left=left_original,
        color=group_colors["Original"],
        edgecolor="white",
        height=0.6,
        label="Original"
    )

    ax.barh(
        y_pos,
        plot_df["Failed_pct"].values,
        left=left_failed,
        color=group_colors["Failed"],
        edgecolor="white",
        height=0.6,
        label="Failed"
    )

    # Percentage labels inside segments
    for i, row in plot_df.iterrows():
        segments = [
            ("Successful_pct", 0),
            ("Original_pct", row["Successful_pct"]),
            ("Failed_pct", row["Successful_pct"] + row["Original_pct"]),
        ]

        for col, left in segments:
            width = row[col]
            if width >= 7:  # label only if wide enough
                ax.text(
                    left + width / 2,
                    i,
                    f"{width:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold"
                )

        # Raw medians to the right of each bar
        raw_text = (
            f"S: {row['Successful_raw']:.4f}   "
            f"O: {row['Original_raw']:.4f}   "
            f"F: {row['Failed_raw']:.4f}"
        )
        ax.text(
            101.5,
            i,
            raw_text,
            va="center",
            ha="left",
            fontsize=8.5,
            color="#444444"
        )

    ax.set_xlim(0, 125)  # leave room for raw values
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 10)], fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Feature"], fontsize=10, fontweight="bold")

    ax.set_title(
        "Feature Comparison of Distance-to-Truthful Baseline",
        fontsize=14,
        fontweight="bold",
        pad=12
    )
    # ax.set_xlabel(
    #     "Relative contribution of each group to feature-level median distance from truthful style",
    #     fontsize=10.5
    # )

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    sns.despine(ax=ax, left=False, bottom=False)

    # Legend
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=10
    )

    # # Explanatory note
    # fig.text(
    #     0.5, 0.01,
    #     "Each feature row sums to 100%. Segment sizes are based on each group's median distance to the truthful baseline for that feature.",
    #     ha="center",
    #     fontsize=10
    # )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = os.path.join(output_dir, "stacked_feature_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

def plot_attacker_victim_heatmap(comp_df: pd.DataFrame, output_dir: str):
    """
    Heatmap:
      Rows = attacker -> victim combinations
      Cols = primary features
      Cell = median improvement toward truthful style for successful attacks:
             OriginalDist - SuccessfulDist
      Positive = successful attacks became more truth-like.
    """
    orig = comp_df[comp_df["group"] == "original"].set_index("sample_id")

    succ = comp_df[comp_df["group"] == "successful"].copy()
    if succ.empty:
        print("No successful attacks found; skipping attacker-victim heatmap.")
        return

    succ["pair"] = succ["attacker"].astype(str) + " → " + succ["victim"].astype(str)

    pair_rows = []
    for pair, df_pair in succ.groupby("pair"):
        row = {"pair": pair}
        for feat in PRIMARY_FEATURES:
            dist_feat = f"{feat}_DistToTruth"

            merged = pd.DataFrame({
                "orig": orig.loc[df_pair["sample_id"].values, dist_feat].values,
                "succ": df_pair[dist_feat].values
            }).dropna()

            if len(merged) > 0:
                # improvement > 0 means closer to truthful
                improvement = np.nanmedian(merged["orig"] - merged["succ"])
            else:
                improvement = np.nan

            row[PRIMARY_LABELS[feat]] = improvement

        pair_rows.append(row)

    heat_df = pd.DataFrame(pair_rows).set_index("pair")
    if heat_df.empty:
        print("Heatmap frame empty; skipping attacker-victim heatmap.")
        return

    plt.figure(figsize=(9, max(3.5, 0.75 * len(heat_df.index) + 1.5)))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Median improvement toward truthful style"}
    )
    plt.title(
        "Attacker → Victim Psycholinguistic Improvement Heatmap\n"
        "(positive = successful attacks became more truth-like)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()

    path = os.path.join(output_dir, "attacker_victim_heatmap.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# -------------------------------------------------------------------
# 10) Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir", default="./liwc_analysis")
    parser.add_argument("--opspam_csv", required=True,
                        help="Path to original OpSpam CSV with truthful + deceptive reviews")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/6] Loading attack results...")
    df = load_results(args.results_dir)

    print("\n[2/6] Loading truthful reference corpus...")
    truth_features = load_truthful_reference(args.opspam_csv)
    truthful_medians = build_truthful_reference_stats(truth_features)

    truth_stats_path = os.path.join(args.output_dir, "truthful_reference_medians.csv")
    pd.DataFrame([truthful_medians]).to_csv(truth_stats_path, index=False)
    print(f"Saved truthful medians to: {truth_stats_path}")

    print("\n[3/6] Building comparison dataframe...")
    comp_df = build_comparison_df(df, truthful_medians)
    comp_path = os.path.join(args.output_dir, "comparison_features.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved comparison features to: {comp_path}")

    print("\n[4/6] Running statistics...")
    stats_df = run_statistics(comp_df)
    stats_path = os.path.join(args.output_dir, "statistical_results.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats to: {stats_path}")

    print("\n[5/6] Generating plots...")
    plot_trajectory_to_truthfulness(comp_df, args.output_dir)
    plot_effect_size_forest(comp_df, args.output_dir)
    plot_attacker_victim_heatmap(comp_df, args.output_dir)

    print("\n[6/6] Done.")
    print("Outputs saved in:", args.output_dir)

if __name__ == "__main__":
    main()