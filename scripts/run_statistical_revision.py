#!/usr/bin/env python3
"""Prespecified revision analyses for the familial erythrocytosis manuscript.

Key safeguards:
* clinical updates are taken from Nico's revised table;
* age, sex, EPO, and thrombosis are excluded from the primary predictor set;
* repeated samples are grouped by participant in every validation split;
* probabilities and performance are summarized once per participant;
* imputation, scaling, feature filtering, and feature selection occur in-fold.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/erythro-matplotlib")

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORKING = ROOT / "working"
OUT = WORKING / "analysis_results"
DATA = WORKING / "analysis_compatibility_full.csv"
META = WORKING / "analysis_metadata.csv"

DEFINED_CODES = ["03-VHL", "03-EGLN1", "04-EPAS1", "05-EPOR", "06-HAH"]
DEFINED_LABELS = ["VHL", "EGLN1", "EPAS1", "EPOR", "HAH"]
LABEL_MAP = {
    "01-Ctrl": "Control",
    "02-Family": "Family",
    "03-VHL": "VHL",
    "03-EGLN1": "EGLN1",
    "04-EPAS1": "EPAS1",
    "05-EPOR": "EPOR",
    "06-HAH": "HAH",
    "07-NDD": "NDD",
    "08-PDE4-associated": "PDE4-associated",
}
COLORS = {
    "Control": "#6E7F80",
    "Family": "#F2A93B",
    "VHL": "#6A3D9A",
    "EGLN1": "#4C78A8",
    "EPAS1": "#1F9EB3",
    "EPOR": "#2DBA61",
    "HAH": "#29A889",
    "NDD": "#9C755F",
    "PDE4-associated": "#C8D800",
}


def clopper_pearson(successes: int, total: int, alpha: float = 0.05):
    low = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return low, high


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


class OmicsTrainingTransformer(BaseEstimator, TransformerMixin):
    """Training-fold detection filter, zero-as-missing rule, and log2 transform."""

    def __init__(self, min_detection: float = 0.35):
        self.min_detection = min_detection

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        frame = frame.mask(frame == 0)
        detection = frame.notna().mean(axis=0)
        self.keep_indices_ = np.flatnonzero(detection.to_numpy() >= self.min_detection)
        kept = frame.iloc[:, self.keep_indices_]
        self.log_mask_ = np.array(
            [
                len(values) > 2 and float(values.min()) > 0
                for _, values in ((col, kept[col].dropna()) for col in kept.columns)
            ],
            dtype=bool,
        )
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        frame = frame.mask(frame == 0).iloc[:, self.keep_indices_]
        arr = frame.to_numpy(dtype=float)
        if arr.size and self.log_mask_.any():
            arr[:, self.log_mask_] = np.log2(arr[:, self.log_mask_])
        arr[~np.isfinite(arr)] = np.nan
        return arr


def load_data():
    data = pd.read_csv(DATA, low_memory=False)
    meta = pd.read_csv(META)
    data["Sample_ID"] = data["Sample_ID"].astype(str)
    meta["Sample_ID"] = meta["Sample_ID"].astype(str)
    merged = data.merge(meta, on=["Sample_ID", "Group"], how="left", validate="one_to_one")
    merged["Group_label_short"] = merged["Group"].map(LABEL_MAP)
    return merged


def block_columns(data):
    clinical_all = [c for c in list(data.columns[2:26]) if c != "Gender"]
    clinical_primary = [c for c in clinical_all if c != "Age"]
    protein = list(data.columns[26:1488])
    metabolites = list(data.columns[1488:1667])
    return clinical_primary, protein, metabolites


def numeric_frame(data, columns):
    return data[columns].apply(pd.to_numeric, errors="coerce")


def make_pipeline_for_block(
    frame: pd.DataFrame,
    clinical_columns: list[str],
    protein_columns: list[str],
    metabolite_columns: list[str],
    k: int,
):
    transformers = []
    if clinical_columns:
        transformers.append(
            (
                "clinical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                clinical_columns,
            )
        )
    if protein_columns:
        transformers.append(
            (
                "proteome",
                Pipeline(
                    [
                        ("raw", OmicsTrainingTransformer(min_detection=0.35)),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                protein_columns,
            )
        )
    if metabolite_columns:
        transformers.append(
            (
                "metabolome",
                Pipeline(
                    [
                        ("raw", OmicsTrainingTransformer(min_detection=0.35)),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                metabolite_columns,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    return Pipeline(
        [
            ("preprocess", pre),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


def participant_grouped_predictions(
    frame: pd.DataFrame,
    clinical_columns: list[str],
    protein_columns: list[str],
    metabolite_columns: list[str],
    k: int,
):
    labels = np.array(DEFINED_LABELS)
    participant_ids = frame["Participant_ID"].astype(str).to_numpy()
    unique_participants = pd.unique(participant_ids)
    rows = []
    for participant in unique_participants:
        test = participant_ids == participant
        train = ~test
        pipe = make_pipeline_for_block(
            frame,
            clinical_columns,
            protein_columns,
            metabolite_columns,
            k=k,
        )
        pipe.fit(frame.loc[train], frame.loc[train, "Group_label_short"])
        probabilities = pipe.predict_proba(frame.loc[test])
        class_order = pipe.named_steps["model"].classes_
        mean_prob = probabilities.mean(axis=0)
        aligned = pd.Series(0.0, index=labels)
        aligned.loc[class_order] = mean_prob
        true_values = frame.loc[test, "Group_label_short"].unique()
        if len(true_values) != 1:
            raise ValueError(f"Participant {participant} spans multiple groups: {true_values}")
        predicted = aligned.idxmax()
        row = {
            "Participant_ID": participant,
            "True": true_values[0],
            "Predicted": predicted,
            "Samples_in_fold": int(test.sum()),
        }
        row.update({f"P_{label}": float(aligned[label]) for label in labels})
        rows.append(row)
    return pd.DataFrame(rows)


def metric_summary(name, predictions):
    truth = predictions["True"]
    pred = predictions["Predicted"]
    correct = int((truth == pred).sum())
    total = int(len(predictions))
    accuracy = accuracy_score(truth, pred)
    balanced = balanced_accuracy_score(truth, pred)
    ci_low, ci_high = clopper_pearson(correct, total, alpha=0.05)
    cm = confusion_matrix(truth, pred, labels=DEFINED_LABELS)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    result = {
        "Model": name,
        "Participants": total,
        "Correct": correct,
        "Accuracy": float(accuracy),
        "Accuracy_95CI_low": float(ci_low),
        "Accuracy_95CI_high": float(ci_high),
        "Balanced_accuracy": float(balanced),
    }
    result.update({f"Recall_{label}": float(value) for label, value in zip(DEFINED_LABELS, recall)})
    return result, cm


def grouped_unknown_probabilities(
    defined: pd.DataFrame,
    unknown: pd.DataFrame,
    block_specs,
):
    participant_tables = []
    for name, clinical, proteins, metabolites, k in block_specs:
        pipe = make_pipeline_for_block(defined, clinical, proteins, metabolites, k=k)
        pipe.fit(defined, defined["Group_label_short"])
        probs = pipe.predict_proba(unknown)
        classes = pipe.named_steps["model"].classes_
        sample = pd.DataFrame(probs, columns=classes)
        sample.insert(0, "Participant_ID", unknown["Participant_ID"].to_numpy())
        part = sample.groupby("Participant_ID", sort=False)[DEFINED_LABELS].mean().reset_index()
        part.insert(1, "Feature_set", name)
        participant_tables.append(part)
    all_prob = pd.concat(participant_tables, ignore_index=True)
    consensus = (
        all_prob.groupby("Participant_ID", sort=False)[DEFINED_LABELS]
        .mean()
        .reset_index()
    )
    consensus["Closest_neighborhood"] = consensus[DEFINED_LABELS].idxmax(axis=1)
    return all_prob, consensus


def descriptive_epo(baseline):
    affected_order = ["VHL", "EGLN1", "EPAS1", "EPOR", "HAH", "NDD", "PDE4-associated"]
    affected = baseline[baseline["Group_label_short"].isin(affected_order)].copy()
    affected["EPO"] = pd.to_numeric(affected["EPO at diagnosis (mIU/mL)"], errors="coerce")
    rows = []
    for group in affected_order:
        values = affected.loc[affected["Group_label_short"] == group, "EPO"].dropna()
        rows.append(
            {
                "Group": group,
                "n": int(len(values)),
                "Median": float(values.median()) if len(values) else np.nan,
                "Q1": float(values.quantile(0.25)) if len(values) else np.nan,
                "Q3": float(values.quantile(0.75)) if len(values) else np.nan,
                "Minimum": float(values.min()) if len(values) else np.nan,
                "Maximum": float(values.max()) if len(values) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    defined = affected[affected["Group_label_short"].isin(DEFINED_LABELS)]
    arrays = [defined.loc[defined["Group_label_short"] == group, "EPO"].dropna() for group in DEFINED_LABELS]
    kw = stats.kruskal(*arrays)
    pairwise_rows = []
    raw_p = []
    for i, group_a in enumerate(DEFINED_LABELS):
        for group_b in DEFINED_LABELS[i + 1 :]:
            a = defined.loc[defined["Group_label_short"] == group_a, "EPO"].dropna()
            b = defined.loc[defined["Group_label_short"] == group_b, "EPO"].dropna()
            test = stats.mannwhitneyu(a, b, alternative="two-sided")
            pairwise_rows.append(
                {
                    "Group_A": group_a,
                    "Group_B": group_b,
                    "n_A": len(a),
                    "n_B": len(b),
                    "U": float(test.statistic),
                    "p_raw": float(test.pvalue),
                }
            )
            raw_p.append(test.pvalue)
    adjusted = benjamini_hochberg(raw_p)
    for row, q in zip(pairwise_rows, adjusted):
        row["q_BH"] = float(q)
    return summary, pd.DataFrame(pairwise_rows), {"H": float(kw.statistic), "p": float(kw.pvalue)}


def thrombosis_summary(baseline):
    affected_order = ["VHL", "EGLN1", "EPAS1", "EPOR", "HAH", "NDD", "PDE4-associated"]
    affected = baseline[baseline["Group_label_short"].isin(affected_order)].copy()
    event = affected["Thrombotic event before age 40"].eq("Yes")
    rows = []
    for group in affected_order:
        mask = affected["Group_label_short"].eq(group)
        n = int(mask.sum())
        events = int((mask & event).sum())
        rows.append(
            {
                "Group": group,
                "Participants": n,
                "Events_before_40": events,
                "Percent": 100 * events / n if n else np.nan,
            }
        )
    vhl = affected["Group_label_short"].eq("VHL")
    table = np.array(
        [
            [int((vhl & event).sum()), int((vhl & ~event).sum())],
            [int((~vhl & event).sum()), int((~vhl & ~event).sum())],
        ]
    )
    fisher = stats.fisher_exact(table, alternative="two-sided")
    return pd.DataFrame(rows), {
        "comparison": "VHL versus all other affected participants",
        "table": table.tolist(),
        "odds_ratio": float(fisher.statistic) if np.isfinite(fisher.statistic) else None,
        "p": float(fisher.pvalue),
    }


def age_sex_tests(baseline_defined, n_perm=50000, seed=1776):
    age_arrays = [
        pd.to_numeric(
            baseline_defined.loc[baseline_defined["Group_label_short"] == group, "Age"],
            errors="coerce",
        ).dropna()
        for group in DEFINED_LABELS
    ]
    kw = stats.kruskal(*age_arrays)
    sex_table = pd.crosstab(baseline_defined["Group_label_short"], baseline_defined["Gender"]).reindex(
        index=DEFINED_LABELS, columns=["F", "M"], fill_value=0
    )
    observed = stats.chi2_contingency(sex_table, correction=False).statistic
    rng = np.random.default_rng(seed)
    groups = baseline_defined["Group_label_short"].to_numpy()
    sexes = baseline_defined["Gender"].to_numpy()
    exceed = 0
    for _ in range(n_perm):
        permuted = rng.permutation(sexes)
        table = pd.crosstab(groups, permuted).reindex(index=DEFINED_LABELS, columns=["F", "M"], fill_value=0)
        stat = stats.chi2_contingency(table, correction=False).statistic
        exceed += stat >= observed - 1e-12
    return {
        "age_kruskal_H": float(kw.statistic),
        "age_kruskal_p": float(kw.pvalue),
        "sex_chi2": float(observed),
        "sex_permutation_p": float((exceed + 1) / (n_perm + 1)),
        "sex_permutations": n_perm,
        "sex_table": sex_table.to_dict(),
    }


def plot_epo(baseline, summary, kw, output):
    order = ["VHL", "EGLN1", "EPAS1", "EPOR", "HAH", "NDD", "PDE4-associated"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    rng = np.random.default_rng(2026)
    values_by_group = []
    for i, group in enumerate(order, start=1):
        values = pd.to_numeric(
            baseline.loc[baseline["Group_label_short"] == group, "EPO at diagnosis (mIU/mL)"],
            errors="coerce",
        ).dropna().to_numpy()
        values_by_group.append(values)
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=36,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            zorder=3,
        )
    bp = ax.boxplot(values_by_group, positions=np.arange(1, len(order) + 1), widths=0.44, showfliers=False, patch_artist=True)
    for patch, group in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.23)
        patch.set_edgecolor(COLORS[group])
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.3)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#666666")
            line.set_linewidth(0.8)
    ax.set_yscale("log")
    ax.set_ylabel("EPO at diagnosis (mIU/mL)")
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(
        [f"{g}\n(n={int(summary.loc[summary.Group == g, 'n'].iloc[0])})" for g in order],
        fontsize=8,
    )
    ax.set_title("Erythropoietin at diagnosis by participant subgroup", fontweight="bold")
    ax.text(0.01, 0.98, f"Defined groups: Kruskal–Wallis p={kw['p']:.3g}", transform=ax.transAxes, va="top", fontsize=8.5)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_thrombosis(summary, fisher, output):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    order = summary["Group"].tolist()
    values = summary["Percent"].to_numpy()
    bars = ax.bar(order, values, color=[COLORS[g] for g in order], edgecolor="white", linewidth=0.6)
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{row.Events_before_40}/{row.Participants}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(35, values.max() + 8))
    ax.set_ylabel("Participants with thrombosis before age 40 (%)")
    ax.set_title("Thrombotic events before age 40", fontweight="bold")
    ax.text(0.01, 0.98, f"VHL vs all other affected groups: Fisher p={fisher['p']:.3g}", transform=ax.transAxes, va="top", fontsize=8.5)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_model_sensitivity(metrics, output):
    display = metrics.copy()
    order = [
        "Age/sex only",
        "Clinical primary",
        "Clinical + age/sex",
        "Clinical + EPO",
        "Metabolome",
        "Proteome",
        "Combined multi-omics",
    ]
    display = display.set_index("Model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y = np.arange(len(display))
    acc = display["Accuracy"].to_numpy()
    low = display["Accuracy_95CI_low"].to_numpy()
    high = display["Accuracy_95CI_high"].to_numpy()
    colors = ["#94A3B8", "#B13A45", "#D97706", "#7C3AED", "#2B9A8A", "#245DA3", "#111827"]
    ax.errorbar(acc, y, xerr=[acc - low, high - acc], fmt="none", ecolor="#64748B", capsize=3, linewidth=1.1)
    ax.scatter(acc, y, s=65, c=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for x, yy, correct, n in zip(acc, y, display["Correct"], display["Participants"]):
        ax.text(min(0.99, x + 0.025), yy, f"{int(correct)}/{int(n)}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(display["Model"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Participant-level accuracy (exact 95% CI)")
    ax.set_title("Participant-grouped leave-one-participant-out validation", fontweight="bold")
    ax.axvline(0.2, color="#CBD5E1", linestyle="--", linewidth=1, label="Five-class chance reference")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    clinical_primary, protein_columns, metabolite_columns = block_columns(data)

    primary = data[data["Primary_omics_analysis"].eq("Yes")].copy()
    baseline = data[data["Baseline_participant_analysis"].eq("Yes")].copy()
    baseline_primary = primary[primary["Baseline_participant_analysis"].eq("Yes")].copy()
    defined = primary[primary["Group"].isin(DEFINED_CODES)].copy()
    defined["Group_label_short"] = defined["Group"].map(LABEL_MAP)
    unknown = primary[primary["Group"].isin(["07-NDD", "08-PDE4-associated"])].copy()

    # Metadata-only covariate frames for prespecified sensitivity models.
    data["Sex_male"] = data["Gender"].map({"F": 0.0, "M": 1.0})
    data["EPO_numeric"] = pd.to_numeric(data["EPO at diagnosis (mIU/mL)"], errors="coerce")
    primary["Sex_male"] = primary["Gender"].map({"F": 0.0, "M": 1.0})
    primary["EPO_numeric"] = pd.to_numeric(primary["EPO at diagnosis (mIU/mL)"], errors="coerce")
    defined["Sex_male"] = defined["Gender"].map({"F": 0.0, "M": 1.0})
    defined["EPO_numeric"] = pd.to_numeric(defined["EPO at diagnosis (mIU/mL)"], errors="coerce")
    unknown["Sex_male"] = unknown["Gender"].map({"F": 0.0, "M": 1.0})
    unknown["EPO_numeric"] = pd.to_numeric(unknown["EPO at diagnosis (mIU/mL)"], errors="coerce")

    block_specs = [
        ("Age/sex only", ["Age", "Sex_male"], [], [], 2),
        ("Clinical primary", clinical_primary, [], [], min(12, len(clinical_primary))),
        ("Clinical + age/sex", clinical_primary + ["Age", "Sex_male"], [], [], min(14, len(clinical_primary) + 2)),
        ("Clinical + EPO", clinical_primary + ["EPO_numeric"], [], [], min(13, len(clinical_primary) + 1)),
        ("Metabolome", [], [], metabolite_columns, 25),
        ("Proteome", [], protein_columns, [], 25),
        ("Combined multi-omics", clinical_primary, protein_columns, metabolite_columns, 35),
    ]

    metric_rows = []
    confusion_outputs = {}
    for name, clinical, proteins, metabolites, k in block_specs:
        predictions = participant_grouped_predictions(defined, clinical, proteins, metabolites, k)
        metric, cm = metric_summary(name, predictions)
        metric_rows.append(metric)
        safe = name.lower().replace(" ", "_").replace("/", "_").replace("+", "plus")
        predictions.to_csv(OUT / f"participant_predictions_{safe}.csv", index=False)
        pd.DataFrame(cm, index=DEFINED_LABELS, columns=DEFINED_LABELS).to_csv(OUT / f"confusion_{safe}.csv")
        confusion_outputs[name] = cm.tolist()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "classifier_metrics.csv", index=False)

    # Only the four non-circular feature sets contribute to unresolved-case consensus.
    unknown_specs = [
        ("Clinical", clinical_primary, [], [], min(12, len(clinical_primary))),
        ("Metabolome", [], [], metabolite_columns, 25),
        ("Proteome", [], protein_columns, [], 25),
        ("Combined", clinical_primary, protein_columns, metabolite_columns, 35),
    ]
    unknown_by_block, unknown_consensus = grouped_unknown_probabilities(defined, unknown, unknown_specs)
    unknown_by_block.to_csv(OUT / "unknown_probabilities_by_feature_set.csv", index=False)
    unknown_consensus.to_csv(OUT / "unknown_consensus_probabilities.csv", index=False)

    epo_summary, epo_pairwise, epo_kw = descriptive_epo(baseline)
    epo_summary.to_csv(OUT / "epo_summary.csv", index=False)
    epo_pairwise.to_csv(OUT / "epo_pairwise_defined_groups.csv", index=False)

    throm_summary, throm_fisher = thrombosis_summary(baseline)
    throm_summary.to_csv(OUT / "thrombosis_summary.csv", index=False)

    baseline_defined = baseline_primary[baseline_primary["Group"].isin(DEFINED_CODES)].copy()
    baseline_defined["Group_label_short"] = baseline_defined["Group"].map(LABEL_MAP)
    confounders = age_sex_tests(baseline_defined)

    plot_epo(baseline, epo_summary, epo_kw, OUT / "epo_distribution.svg")
    plot_thrombosis(throm_summary, throm_fisher, OUT / "thrombosis_before_40.svg")
    plot_model_sensitivity(metrics, OUT / "participant_grouped_model_sensitivity.svg")

    summary = {
        "counts": {
            "input_samples": int(len(data)),
            "input_participants": int(data["Participant_ID"].nunique()),
            "primary_omics_samples": int(len(primary)),
            "primary_omics_participants": int(primary["Participant_ID"].nunique()),
            "baseline_primary_participants": int(len(baseline_primary)),
            "defined_training_samples": int(len(defined)),
            "defined_training_participants": int(defined["Participant_ID"].nunique()),
            "undefined_samples": int(len(unknown)),
            "undefined_participants": int(unknown["Participant_ID"].nunique()),
        },
        "epo_defined_kruskal": epo_kw,
        "thrombosis": throm_fisher,
        "age_sex": confounders,
        "classifier_metrics": metric_rows,
        "confusion_matrices": confusion_outputs,
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
