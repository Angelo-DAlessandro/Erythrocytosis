#!/usr/bin/env python3
"""Generate revised Supplementary Figures 1-5 for the erythrocytosis report.

The script deliberately separates sample-level descriptive omics displays from
participant-level inferential and validation analyses. Age, sex, EPO and
thrombosis are excluded from the primary predictor matrix; EPO/thrombosis are
shown only in prespecified secondary clinical panels.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

GROUP_MAP = {
    "01-Ctrl": "Control",
    "02-Family": "Family",
    "03-VHL": "VHL",
    "03-EGLN1": "EGLN1",
    "04-EPAS1": "EPAS1",
    "05-EPOR": "EPOR",
    "06-HAH": "HAH",
    "07-NDD": "Unresolved",
    "08-PDE4-associated": "PDE4 family",
}
ORDER = ["Control", "Family", "VHL", "EGLN1", "EPAS1", "EPOR", "HAH", "Unresolved", "PDE4 family"]
DEFINED = ["VHL", "EGLN1", "EPAS1", "EPOR", "HAH"]
COLORS = {
    "Control": "#636363",
    "Family": "#D9911B",
    "VHL": "#7650A3",
    "EGLN1": "#347BB7",
    "EPAS1": "#119E9A",
    "EPOR": "#35A85D",
    "HAH": "#187D68",
    "Unresolved": "#956650",
    "PDE4 family": "#AEBB00",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--merged", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def panel(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", clip_on=False)


def tidy_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def find_col(df: pd.DataFrame, term: str) -> str | None:
    if term in df.columns:
        return term
    low = term.lower()
    exact = [c for c in df.columns if c.lower() == low]
    if exact:
        return exact[0]
    contains = [c for c in df.columns if low in c.lower()]
    return contains[0] if contains else None


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def prep_omics(frame: pd.DataFrame, cols: list[str], log: bool = True) -> pd.DataFrame:
    x = frame[cols].apply(pd.to_numeric, errors="coerce").mask(lambda z: z == 0)
    if log:
        x = np.log10(x.clip(lower=1e-12))
    # Some raw features can be absent in a subgroup-filtered frame. Remove them
    # explicitly because newer scikit-learn versions otherwise silently drop
    # empty columns during imputation and break the column mapping.
    x = x.loc[:, x.notna().any(axis=0)]
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=x.columns, index=frame.index)
    keep = x.std(axis=0, ddof=0) > 1e-12
    x = x.loc[:, keep]
    return pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns, index=x.index)


def prep_clinical(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].apply(pd.to_numeric, errors="coerce")
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=cols, index=frame.index)
    keep = x.std(axis=0, ddof=0) > 1e-12
    x = x.loc[:, keep]
    return pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns, index=x.index)


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def count_label(meta: pd.DataFrame, group: str, primary_only: bool = False) -> str:
    m = meta[meta["Group_short"] == group]
    if primary_only:
        m = m[m["Primary_omics_analysis"].eq("Yes")]
    return f"n={len(m)}; N={m['Participant_ID'].nunique()}"


def generate_figure1(df, meta, clinical_cols, protein_cols, metab_cols, outdir):
    primary = meta["Primary_omics_analysis"].eq("Yes").to_numpy()
    xclin = prep_clinical(df, clinical_cols)
    xprot = prep_omics(df, protein_cols)
    xmet = prep_omics(df, metab_cols)
    # Equal layer contribution for unsupervised QC: cap each layer by variance.
    psel = xprot.var().nlargest(175).index
    msel = xmet.var().nlargest(175).index
    qall = pd.concat([xclin, xprot[psel], xmet[msel]], axis=1)

    fig = plt.figure(figsize=(17, 10.8))
    gs = GridSpec(2, 4, figure=fig, wspace=0.38, hspace=0.38)
    fig.suptitle("Supplementary Figure 1. Cohort accounting and unsupervised quality control", fontsize=16, fontweight="bold", y=0.988)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A"); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(FancyBboxPatch((0.25, 7.65), 9.5, 1.55, boxstyle="round,pad=.08", fc="#F3F5F7", ec="#9BA3AA"))
    ax.text(5, 8.65, "Reconciled cohort", ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 8.12, "65 samples from 63 participants", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((5, 7.6), (5, 6.9), arrowstyle="-|>", mutation_scale=15, color="#6B7278"))
    ax.add_patch(FancyBboxPatch((0.25, 5.35), 9.5, 1.45, boxstyle="round,pad=.08", fc="#FFF3F1", ec="#C84E43"))
    ax.text(5, 6.3, "Primary omics exclusions", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(5, 5.75, "F2_1 and F6_1 (two Family participants)", ha="center", fontsize=8.5)
    ax.add_patch(FancyArrowPatch((5, 5.3), (5, 4.6), arrowstyle="-|>", mutation_scale=15, color="#6B7278"))
    ax.add_patch(FancyBboxPatch((0.25, 2.6), 9.5, 1.85, boxstyle="round,pad=.08", fc="#EFF8F5", ec="#408A72"))
    ax.text(5, 3.9, "Primary omics cohort", ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 3.35, "63 samples from 61 participants", ha="center", fontsize=9)
    ax.text(5, 2.9, "Defined-class training: 40 samples / 39 participants", ha="center", fontsize=8)
    ax.text(5, 1.86, "Repeated samples: P14/P14MED and P47/P47MED\nwere grouped by participant in validation", ha="center", fontsize=8)

    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    omics_raw = pd.concat([
        df[clinical_cols].apply(pd.to_numeric, errors="coerce"),
        df[protein_cols].apply(pd.to_numeric, errors="coerce").mask(lambda z: z == 0),
        df[metab_cols].apply(pd.to_numeric, errors="coerce").mask(lambda z: z == 0),
    ], axis=1)
    miss = omics_raw.isna().mean(axis=1)
    idx = np.argsort(miss.to_numpy())
    ranks = np.arange(1, len(df) + 1)
    c = ["#C63D3D" if not primary[i] else COLORS[meta.iloc[i]["Group_short"]] for i in idx]
    ax.scatter(ranks, miss.iloc[idx], c=c, s=28, edgecolor="white", linewidth=0.35)
    ax.set_title("Sample-level missingness", fontsize=9, fontweight="bold")
    ax.set_xlabel("Samples ranked by missing fraction", fontsize=8); ax.set_ylabel("Fraction missing", fontsize=8)
    ax.text(0.04, 0.96, f"Primary cohort median: {miss[primary].median():.3f}\nExcluded samples shown in red", transform=ax.transAxes, va="top", fontsize=7, bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))
    tidy_axes(ax)

    ax = fig.add_subplot(gs[0, 2]); panel(ax, "C")
    layer_raw = [df[clinical_cols].apply(pd.to_numeric, errors="coerce"), df[protein_cols].apply(pd.to_numeric, errors="coerce").mask(lambda z: z == 0), df[metab_cols].apply(pd.to_numeric, errors="coerce").mask(lambda z: z == 0)]
    vals = [1 - z.isna().mean().mean() for z in layer_raw]
    layer_names = ["Clinical", "Proteome", "Metabolome"]
    layer_colors = ["#555555", "#B5434C", "#238E80"]
    bars = ax.bar(layer_names, vals, color=layer_colors, edgecolor="white")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Proportion observed", fontsize=8); ax.set_title("Data-layer completeness", fontsize=9, fontweight="bold")
    for b, v, nfeat in zip(bars, vals, [len(clinical_cols), len(protein_cols), len(metab_cols)]):
        ax.text(b.get_x() + b.get_width()/2, v + .025, f"{v:.2f}\n{nfeat:,} variables", ha="center", fontsize=7)
    ax.tick_params(axis="x", labelsize=7); tidy_axes(ax)

    ax = fig.add_subplot(gs[0, 3]); panel(ax, "D")
    feat_miss = [z.isna().mean().to_numpy() + 1e-5 for z in layer_raw]
    vp = ax.violinplot(feat_miss, showextrema=False, showmedians=True, widths=.78)
    for body, color in zip(vp["bodies"], layer_colors):
        body.set_facecolor(color); body.set_alpha(.82); body.set_edgecolor("#333333")
    ax.set_yscale("log"); ax.set_xticks([1,2,3]); ax.set_xticklabels(layer_names, fontsize=7)
    ax.set_ylabel("Per-variable missing fraction", fontsize=8); ax.set_title("Variable-level missingness", fontsize=9, fontweight="bold")
    tidy_axes(ax)

    ax = fig.add_subplot(gs[1, 0]); panel(ax, "E")
    q = qall.loc[primary]
    vars_ = q.var().nlargest(min(350, q.shape[1])).index
    cor = np.corrcoef(q[vars_].to_numpy())
    dist = np.maximum(1 - cor, 0); np.fill_diagonal(dist, 0)
    try: leaves = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    except Exception: leaves = np.arange(len(cor))
    im = ax.imshow(cor[np.ix_(leaves, leaves)], cmap="RdBu_r", vmin=.2, vmax=1, interpolation="nearest")
    ax.set_title("Sample correlation (primary cohort)", fontsize=9, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=.046, pad=.03, label="Pearson r")

    for grid, mask, title, letter in [(gs[1,1], np.ones(len(df), dtype=bool), "PCA before exclusions", "F"), (gs[1,2], primary, "PCA: primary omics cohort", "G")]:
        ax = fig.add_subplot(grid); panel(ax, letter)
        scores = PCA(n_components=2, random_state=7).fit_transform(qall.loc[mask])
        md = meta.loc[mask].reset_index(drop=True)
        for g in ORDER:
            gm = md["Group_short"].eq(g).to_numpy()
            if gm.any(): ax.scatter(scores[gm,0], scores[gm,1], s=35, c=COLORS[g], edgecolor="white", linewidth=.4, label=g)
        if letter == "F":
            exc = ~primary
            ax.scatter(scores[exc,0], scores[exc,1], marker="X", s=90, c="#C63D3D", edgecolor="#222222", linewidth=.5, label="Excluded")
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8); ax.set_title(title, fontsize=9, fontweight="bold"); tidy_axes(ax)
        if letter == "G": ax.legend(fontsize=5.7, frameon=False, ncol=2, loc="best")

    ax = fig.add_subplot(gs[1, 3]); panel(ax, "H")
    scores = PCA(n_components=2, random_state=7).fit_transform(qall.loc[primary])
    md = meta.loc[primary].reset_index(drop=True)
    labs, centroids = [], []
    for g in ORDER:
        gm = md["Group_short"].eq(g).to_numpy()
        if gm.any(): labs.append(g); centroids.append(scores[gm].mean(axis=0))
    centroids = np.asarray(centroids)
    d = np.sqrt(((centroids[:,None,:] - centroids[None,:,:])**2).sum(axis=2))
    im = ax.imshow(d, cmap="viridis")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6)
    ax.set_title("Distances between PCA centroids", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.text(.5, .008, "n = samples; N = participants. PCA was used for unsupervised quality control; no outcome labels entered the projection.", ha="center", fontsize=7.5)
    save(fig, outdir, "Supplementary_Figure_1_revised")


def volcano(frame, cols, group, group_series):
    x = prep_omics(frame, cols)
    gm = group_series.eq(group).to_numpy()
    rows = []
    for c in x.columns:
        a, b = x.loc[gm, c], x.loc[~gm, c]
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue if gm.sum() > 1 else 1.0
        rows.append((c, float(np.median(a)-np.median(b)), p))
    r = pd.DataFrame(rows, columns=["feature","difference","p"])
    r["q"] = bh_adjust(r["p"].to_numpy()); r["mlogp"] = -np.log10(r["p"].clip(1e-300))
    r["sig"] = r["q"].lt(.05) & r["difference"].abs().ge(.8)
    return r


def volcano_plot(ax, r, title, max_labels=6):
    sigup = r["sig"] & r["difference"].gt(0); sigdn = r["sig"] & r["difference"].lt(0)
    ax.scatter(r.loc[~r["sig"],"difference"], r.loc[~r["sig"],"mlogp"], s=7, c="#BDBDBD", alpha=.65)
    ax.scatter(r.loc[sigdn,"difference"], r.loc[sigdn,"mlogp"], s=12, c="#2366A2")
    ax.scatter(r.loc[sigup,"difference"], r.loc[sigup,"mlogp"], s=12, c="#B73743")
    ax.axvline(0, ls="--", lw=.6, color="#555555")
    labels = r.sort_values("p").head(max_labels)
    for _, z in labels.iterrows():
        ax.annotate(str(z.feature)[:23], (z.difference, z.mlogp), xytext=(3 if z.difference>=0 else -3, 3), textcoords="offset points", ha="left" if z.difference>=0 else "right", fontsize=5.5)
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_xlabel("Median difference (z score)", fontsize=7); ax.set_ylabel("−log10(P)", fontsize=7)
    ax.text(.98,.96,f"FDR-significant: ↑{sigup.sum()}  ↓{sigdn.sum()}",transform=ax.transAxes,ha="right",va="top",fontsize=6.3)
    tidy_axes(ax)


def generate_figure2(df, meta, protein_cols, metab_cols, outdir):
    primary = meta["Primary_omics_analysis"].eq("Yes").to_numpy()
    d = df.loc[primary].reset_index(drop=True); g = meta.loc[primary,"Group_short"].reset_index(drop=True)
    panels = [(x, typ, cols) for x in ["VHL","EPOR","HAH","Unresolved"] for typ, cols in [("proteome",protein_cols),("metabolome",metab_cols)]]
    fig = plt.figure(figsize=(16.8,10.7)); gs = GridSpec(3,3,figure=fig,wspace=.27,hspace=.34)
    fig.suptitle("Supplementary Figure 2. Subgroup-associated red-cell omics features",fontsize=16,fontweight="bold",y=.988)
    summaries=[]
    for i,(grp,typ,cols) in enumerate(panels):
        ax=fig.add_subplot(gs[i//3,i%3]); panel(ax,chr(65+i)); r=volcano(d,cols,grp,g); volcano_plot(ax,r,f"{grp}: {typ}")
        summaries.append((grp,typ,int((r.sig&r.difference.gt(0)).sum()),int((r.sig&r.difference.lt(0)).sum())))
    ax=fig.add_subplot(gs[2,2]); panel(ax,"I"); ax.axis("off")
    ax.set_title("FDR-significant feature counts",fontsize=10,fontweight="bold",pad=8)
    tab=pd.DataFrame(summaries,columns=["Group","Layer","Increased","Decreased"])
    rows=[]
    for grp in ["VHL","EPOR","HAH","Unresolved"]:
        for typ in ["proteome","metabolome"]:
            z=tab[(tab.Group==grp)&(tab.Layer==typ)].iloc[0]
            rows.append([grp,typ.capitalize(),z.Increased,z.Decreased])
    table=ax.table(cellText=rows,colLabels=["Group","Layer","↑","↓"],loc="center",cellLoc="center",colLoc="center",bbox=[.05,.16,.9,.72])
    table.auto_set_font_size(False); table.set_fontsize(7.5)
    for (r,c),cell in table.get_celld().items():
        if r==0: cell.set_facecolor("#E9EEF2"); cell.set_text_props(fontweight="bold")
        elif r%2==0: cell.set_facecolor("#F7F7F7")
    fig.text(.5,.01,"Mann–Whitney U tests compare each subgroup with the remaining primary omics cohort; Benjamini–Hochberg q<0.05 and |median z-score difference|≥0.8 define highlighted features.",ha="center",fontsize=7.2)
    save(fig,outdir,"Supplementary_Figure_2_revised")


def violin_panel(ax, values, groups, order, colors, title, ylabel, log=False):
    arrays=[]; used=[]
    for g in order:
        a=pd.to_numeric(values[groups.eq(g)],errors="coerce")
        if log:
            # Zero denotes non-detection in the raw omics matrix, not true zero.
            a=a.mask(a<=0)
        a=a.dropna().to_numpy()
        if log: a=np.log10(a)
        if len(a): arrays.append(a); used.append(g)
    parts=ax.violinplot(arrays,positions=np.arange(1,len(used)+1),showextrema=False,widths=.78)
    for body,g in zip(parts["bodies"],used): body.set_facecolor(colors[g]); body.set_edgecolor("#222222"); body.set_alpha(.7)
    ax.boxplot(arrays,positions=np.arange(1,len(used)+1),widths=.22,showfliers=False,patch_artist=True,boxprops=dict(facecolor="white",alpha=.8),medianprops=dict(color="black"))
    rng=np.random.default_rng(15)
    for i,(a,g) in enumerate(zip(arrays,used),1): ax.scatter(i+rng.normal(0,.055,len(a)),a,s=12,c=colors[g],edgecolor="white",linewidth=.3,zorder=4)
    test=[a for a in arrays if len(a)>1]
    p=kruskal(*test).pvalue if len(test)>1 else np.nan
    ax.set_title(f"{title}\nKruskal–Wallis P={p:.2g}" if np.isfinite(p) else title,fontsize=8.5,fontweight="bold")
    ax.set_ylabel(ylabel,fontsize=7); ax.set_xticks(range(1,len(used)+1)); ax.set_xticklabels(used,rotation=40,ha="right",fontsize=5.7)
    tidy_axes(ax)


def generate_figure3(df, meta, merged, summary, protein_cols, metab_cols, outdir):
    base=meta["Baseline_participant_analysis"].eq("Yes").to_numpy()
    d=df.loc[base].reset_index(drop=True); md=meta.loc[base].reset_index(drop=True); mm=merged.loc[base].reset_index(drop=True)
    fig=plt.figure(figsize=(17,13)); gs=GridSpec(4,3,figure=fig,wspace=.34,hspace=.62)
    fig.suptitle("Supplementary Figure 3. Clinical context and representative marker distributions",fontsize=16,fontweight="bold",y=.992)
    marker_specs=[
        ("Emogas pHv","Venous pH","Venous pH",False),
        ("vLac (mmol/L)","Venous lactate","mmol/L",False),
        ("IST (%)","Transferrin saturation","%",False),
        ("Nicotinamide","Nicotinamide","log10 intensity",True),
        ("Kynurenine","Kynurenine","log10 intensity",True),
        ("acyl-C4-OH","3-hydroxybutyrylcarnitine (acyl-C4-OH)","log10 intensity",True),
        ("RAB35","RAB35","log10 abundance",True),
        ("HPRT1","HPRT1","log10 abundance",True),
        ("GYPC","Glycophorin C (GYPC)","log10 abundance",True),
    ]
    for i,(term,title,ylabel,log) in enumerate(marker_specs):
        ax=fig.add_subplot(gs[i//3,i%3]); panel(ax,chr(65+i)); c=find_col(d,term)
        violin_panel(ax,d[c],md.Group_short,ORDER,COLORS,title,ylabel,log)
    ax=fig.add_subplot(gs[3,0]); panel(ax,"J")
    epo_order=["VHL","EGLN1","EPAS1","EPOR","HAH","Unresolved","PDE4 family"]
    violin_panel(ax,md["EPO at diagnosis (mIU/mL)"],md.Group_short,epo_order,COLORS,"Erythropoietin at diagnosis","log10 EPO (mIU/mL)",True)
    ax.text(.02,-.42,"EPO was evaluated secondarily and was not included in the primary clinical predictor set.",transform=ax.transAxes,fontsize=6.3)

    ax=fig.add_subplot(gs[3,1]); panel(ax,"K")
    affected=md[md.Group_short.isin(epo_order)].copy()
    rates=[]; ns=[]
    for g in epo_order:
        z=affected[affected.Group_short.eq(g)]
        raw=z["Thrombotic event before age 40"]
        v=raw.map({"Yes":1,"No":0,"yes":1,"no":0,True:1,False:0,1:1,0:0})
        ns.append(v.notna().sum()); rates.append(100*v.mean() if v.notna().any() else np.nan)
    bars=ax.bar(range(len(epo_order)),rates,color=[COLORS[g] for g in epo_order],edgecolor="white")
    ax.set_xticks(range(len(epo_order))); ax.set_xticklabels(epo_order,rotation=40,ha="right",fontsize=5.7); ax.set_ylabel("Participants with event (%)",fontsize=7)
    ax.set_title(f"Thrombosis before age 40\nVHL vs other affected: Fisher P={summary['thrombosis']['p']:.3g}",fontsize=8.5,fontweight="bold")
    ax.set_ylim(0,max(35,np.nanmax(rates)*1.22)); tidy_axes(ax)
    for b,r,n in zip(bars,rates,ns):
        if np.isfinite(r): ax.text(b.get_x()+b.get_width()/2,r+1,f"{r:.0f}%\nN={n}",ha="center",fontsize=5.8)

    ax=fig.add_subplot(gs[3,2]); panel(ax,"L"); ax.axis("off")
    agep=summary["age_sex"]["age_kruskal_p"]; sexp=summary["age_sex"]["sex_permutation_p"]
    ax.add_patch(FancyBboxPatch((.04,.55),.92,.34,boxstyle="round,pad=.04",fc="#F2F5F7",ec="#99A4AD"))
    ax.text(.5,.82,"Confounding checks in defined groups",ha="center",fontsize=9,fontweight="bold")
    ax.text(.5,.69,f"Age: Kruskal–Wallis P={agep:.3f}\nSex distribution: 50,000-permutation P={sexp:.3f}",ha="center",va="center",fontsize=8)
    ax.add_patch(FancyBboxPatch((.04,.10),.92,.31,boxstyle="round,pad=.04",fc="#EFF8F5",ec="#5A9A84"))
    ax.text(.5,.34,"Model specification",ha="center",fontsize=9,fontweight="bold")
    ax.text(.5,.21,"Age, sex, EPO and thrombosis were excluded\nfrom the primary clinical classifier.",ha="center",va="center",fontsize=8)
    fig.text(.5,.008,"Continuous panels show one baseline observation per participant. Points, density envelopes, and embedded boxplots are descriptive; global P values are unadjusted.",ha="center",fontsize=7.2)
    save(fig,outdir,"Supplementary_Figure_3_revised")


def confusion_panel(ax, cm, classes, title, accuracy, bacc):
    cm=np.asarray(cm,dtype=int); row=100*cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
    im=ax.imshow(row,cmap="Blues",vmin=0,vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,f"{cm[i,j]}\n{row[i,j]:.0f}%" if cm[i,j] else "·",ha="center",va="center",fontsize=6.2,color="white" if row[i,j]>55 else "#222222")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes,rotation=40,ha="right",fontsize=6)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes,fontsize=6)
    ax.set_xlabel("Predicted",fontsize=7); ax.set_ylabel("Observed",fontsize=7)
    ax.set_title(f"{title}\nAccuracy {accuracy:.2f}; balanced {bacc:.2f}",fontsize=8,fontweight="bold")
    return im


def generate_figure4(df, meta, merged, results_dir, protein_cols, metab_cols, outdir):
    metrics=pd.read_csv(results_dir/"classifier_metrics.csv")
    model_map=[("Clinical primary","Clinical variables"),("Metabolome","Metabolome"),("Proteome","Proteome"),("Combined multi-omics","Combined multi-omics")]
    fig=plt.figure(figsize=(17.5,10.8)); gs=GridSpec(2,4,figure=fig,wspace=.42,hspace=.42,height_ratios=[.95,1.12])
    fig.suptitle("Supplementary Figure 4. Participant-grouped validation and sensitivity analyses",fontsize=16,fontweight="bold",y=.988)
    last=None
    for i,(model,title) in enumerate(model_map):
        ax=fig.add_subplot(gs[0,i]); panel(ax,chr(65+i),x=-.17,y=1.1)
        row=metrics[metrics.Model.eq(model)].iloc[0]
        stem={"Clinical primary":"clinical_primary","Metabolome":"metabolome","Proteome":"proteome","Combined multi-omics":"combined_multi-omics"}[model]
        cm=pd.read_csv(results_dir/f"confusion_{stem}.csv",index_col=0).loc[DEFINED,DEFINED].to_numpy()
        last=confusion_panel(ax,cm,DEFINED,f"Leave-one-participant-out: {title}",row.Accuracy,row.Balanced_accuracy)
    cax=fig.add_axes([.944,.605,.009,.245]); cb=fig.colorbar(last,cax=cax); cb.set_label("Row %",fontsize=7); cb.ax.tick_params(labelsize=6)

    ax=fig.add_subplot(gs[1,0]); panel(ax,"E")
    show_order=["Age/sex only","Clinical primary","Clinical + age/sex","Clinical + EPO","Metabolome","Proteome","Combined multi-omics"]
    mm=metrics.set_index("Model").loc[show_order].reset_index()
    ypos=np.arange(len(mm))[::-1]
    ax.errorbar(mm.Accuracy,ypos,xerr=[mm.Accuracy-mm.Accuracy_95CI_low,mm.Accuracy_95CI_high-mm.Accuracy],fmt="o",color="#245F8D",ecolor="#7B9DB6",capsize=3)
    ax.set_yticks(ypos); ax.set_yticklabels(mm.Model,fontsize=6.4); ax.axvline(.2,ls="--",lw=.7,color="#777777")
    ax.set_xlim(0,1.02); ax.set_xlabel("Participant-level accuracy (95% exact CI)",fontsize=7); ax.set_title("Predictor-set sensitivity",fontsize=9,fontweight="bold"); tidy_axes(ax)

    # Compact age-free clinical–omics correlation network.
    ax=fig.add_subplot(gs[1,1:3]); panel(ax,"F"); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    base=meta["Baseline_participant_analysis"].eq("Yes") & meta.Group_short.isin(DEFINED)
    d=df.loc[base].reset_index(drop=True)
    clin_terms=["Glucose (mg/dL)","vLac (mmol/L)","Emogas pHv","IST (%)","Uric A (mg/dL)","PLT (/microL)"]
    met_terms=["Hypoxanthine","Nicotinamide","Kynurenine","L-alanine","Lactate","Ornithine"]
    prot_terms=["RAB35","HPRT1","GYPC","NDUFS1","S100A8","NME1"]
    clin=[find_col(d,x) for x in clin_terms]; met=[find_col(d,x) for x in met_terms]; prot=[find_col(d,x) for x in prot_terms]
    clin=[x for x in clin if x]; met=[x for x in met if x]; prot=[x for x in prot if x]
    raw=d[clin+met+prot].apply(pd.to_numeric,errors="coerce")
    for c in met+prot: raw[c]=np.log10(raw[c].mask(raw[c].eq(0)).clip(lower=1e-12))
    raw=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(raw),columns=raw.columns)
    edges=[]
    for a in clin:
        for b in met+prot:
            rho,p=spearmanr(raw[a],raw[b]);
            if np.isfinite(rho) and abs(rho)>=.35: edges.append((a,b,rho))
    edges=sorted(edges,key=lambda z:abs(z[2]),reverse=True)[:40]
    pos={}
    for i,c in enumerate(clin): pos[c]=(.09,.86-i*.72/max(1,len(clin)-1))
    for i,c in enumerate(met): pos[c]=(.91,.86-i*.72/max(1,len(met)-1))
    for i,c in enumerate(prot): pos[c]=(.5,.12+i*.43/max(1,len(prot)-1))
    for a,b,rho in edges:
        if a in pos and b in pos:
            ax.add_patch(FancyArrowPatch(pos[a],pos[b],arrowstyle="-",connectionstyle="arc3,rad=.12",lw=.5+2*abs(rho),color=plt.cm.RdBu_r((rho+1)/2),alpha=.52))
    for cols,color in [(clin,"#B5434C"),(met,"#238E80"),(prot,"#2D69A3")]:
        for c in cols:
            x,y=pos[c]; ax.scatter(x,y,s=75,c=color,edgecolor="white",zorder=3)
            ha="right" if x<.2 else "left" if x>.8 else "center"; dx=-.018 if x<.2 else .018 if x>.8 else 0; dy=0 if x!=.5 else -.035
            ax.text(x+dx,y+dy,c.replace("Ferritina (mg/dL)","Ferritin (ng/mL)").replace("Ferritina","Ferritin")[:22],ha=ha,va="center" if x!=.5 else "top",fontsize=6)
    ax.text(.09,.98,"Clinical",ha="center",color="#B5434C",fontweight="bold",fontsize=9)
    ax.text(.91,.98,"Metabolites",ha="center",color="#238E80",fontweight="bold",fontsize=9)
    ax.text(.5,.03,"Proteins",ha="center",color="#2D69A3",fontweight="bold",fontsize=9)
    ax.set_title("Age-free clinical–omics Spearman network (defined participants)",fontsize=9,fontweight="bold")

    sub=gs[1,3].subgridspec(2,1,hspace=.52)
    pairs=[("Emogas pHv","vLac (mmol/L)","Venous pH","Venous lactate"),("Ferritin (ng/mL)","IST (%)","Ferritin (ng/mL)","Transferrin saturation")]
    for j,(xc,yc,xlab,ylab) in enumerate(pairs):
        ax=fig.add_subplot(sub[j,0]); panel(ax,chr(71+j),x=-.17,y=1.14)
        x=pd.to_numeric(d[xc],errors="coerce"); y=pd.to_numeric(d[yc],errors="coerce"); ok=x.notna()&y.notna()
        groups=meta.loc[base,"Group_short"].reset_index(drop=True)
        for g in DEFINED:
            m=ok&groups.eq(g); ax.scatter(x[m],y[m],s=24,c=COLORS[g],edgecolor="white",linewidth=.35)
        rho,p=spearmanr(x[ok],y[ok]);
        if ok.sum()>2:
            coef=np.polyfit(x[ok],y[ok],1); xx=np.linspace(x[ok].min(),x[ok].max(),100); ax.plot(xx,coef[0]*xx+coef[1],color="#333333",lw=1)
        ax.set_title(f"{xlab} vs {ylab}\nSpearman ρ={rho:.2f}, P={p:.2g}",fontsize=7.8,fontweight="bold"); ax.set_xlabel(xlab,fontsize=6.5); ax.set_ylabel(ylab,fontsize=6.5); tidy_axes(ax)
    fig.text(.5,.009,"All preprocessing, filtering and feature selection were refit inside each training fold. Repeated samples were withheld together and their probabilities averaged to one participant-level prediction.",ha="center",fontsize=7.2)
    save(fig,outdir,"Supplementary_Figure_4_revised")


def box(ax, xy, wh, text, fc, ec="#59636B", fontsize=7.2, weight="normal"):
    x,y=xy; w,h=wh
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.02,rounding_size=.025",fc=fc,ec=ec,lw=.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,fontweight=weight)


def generate_figure5(outdir):
    fig=plt.figure(figsize=(17,10.7)); gs=GridSpec(2,2,figure=fig,wspace=.16,hspace=.24)
    fig.suptitle("Supplementary Figure 5. Framework for omics-guided prioritization in unresolved erythrocytosis",fontsize=16,fontweight="bold",y=.988)
    ax=fig.add_subplot(gs[0,0]); panel(ax,"A"); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Participant-level workflow",fontsize=10,fontweight="bold")
    items=[("Unresolved erythrocytosis\nparticipant",.82,"#F3EDE9"),("Clinical phenotype + RBC\nproteome + metabolome",.61,"#EAF2F7"),("Deviation scores and\npathway-module mapping",.40,"#EAF7F2"),("Defined molecular neighborhoods\nVHL · EGLN1 · EPAS1 · EPOR · HAH",.19,"#EEEAF6")]
    for i,(txt,y,fc) in enumerate(items):
        box(ax,(.18,y),(.64,.13),txt,fc,fontsize=8.3,weight="bold" if i in [0,3] else "normal")
        if i<len(items)-1: ax.add_patch(FancyArrowPatch((.5,y),(.5,y-.075),arrowstyle="-|>",mutation_scale=13,color="#68727A"))
    ax.text(.5,.07,"Output: candidates for genomic reanalysis—not a diagnostic assignment",ha="center",fontsize=8,color="#8A3030",fontweight="bold")

    ax=fig.add_subplot(gs[0,1]); panel(ax,"B"); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Biological routes used to interpret deviations",fontsize=10,fontweight="bold")
    routes=[("Oxygen sensing","VHL · EGLN1 · EPAS1"),("Erythropoietic signaling","EPOR · JAK2/STAT5"),("Oxygen delivery / RBC metabolism","High-affinity Hb · 2,3-BPG"),("Membrane and survival","Cytoskeleton · ion transport"),("Maturation and trafficking","Vesicles · organelle clearance"),("Redox and proteostasis","Glutathione · proteasome")]
    for i,(title,desc) in enumerate(routes):
        x=.06+(i%2)*.48; y=.72-(i//2)*.27
        box(ax,(x,y),(.40,.18),f"{title}\n{desc}","#F5F7F8",fontsize=7.6,weight="bold" if i<2 else "normal")

    ax=fig.add_subplot(gs[1,0]); panel(ax,"C"); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Illustrative candidate-prioritization score",fontsize=10,fontweight="bold")
    components=[("Protein deviation",.28,"#B5434C"),("Pathway/module deviation",.24,"#238E80"),("Phenotypic plausibility",.20,"#7B5AA6"),("Rare-variant evidence",.18,"#D08B27"),("Erythroid expression/function",.10,"#497A9F")]
    y=.78
    for label,w,color in components:
        ax.barh(y,w,left=.16,color=color,height=.105); ax.text(.14,y,label,ha="right",va="center",fontsize=7.4); ax.text(.16+w+.015,y,f"weight {w:.2f}",va="center",fontsize=7)
        y-=.145
    ax.text(.5,.08,"Weights are conceptual and require prespecification and external validation.",ha="center",fontsize=8,color="#8A3030")

    ax=fig.add_subplot(gs[1,1]); panel(ax,"D"); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Illustrative participant-level output",fontsize=10,fontweight="bold")
    rows=[("Participant","Closest neighborhood","Candidate route"),("P38","VHL","Oxygen sensing"),("P40","HAH","Oxygen delivery"),("P47","EPOR","Erythropoietic signaling"),("P50","EGLN1 / EPOR","Mixed; prioritize reanalysis")]
    table=ax.table(cellText=rows[1:],colLabels=rows[0],cellLoc="center",colLoc="center",bbox=[.04,.30,.92,.58],colWidths=[.20,.34,.46])
    table.auto_set_font_size(False); table.set_fontsize(7.4)
    for (r,c),cell in table.get_celld().items():
        if r==0: cell.set_facecolor("#DCE6ED"); cell.set_text_props(fontweight="bold")
        elif r%2==0: cell.set_facecolor("#F5F7F8")
    ax.text(.5,.17,"Neighborhood probabilities are exploratory and participant-averaged.\nThey nominate hypotheses for WES/WGS review; they do not establish causality.",ha="center",fontsize=8)
    save(fig,outdir,"Supplementary_Figure_5_revised")


def main():
    a=parse_args(); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.data); meta=pd.read_csv(a.metadata); merged=pd.read_csv(a.merged)
    # Align the three inputs explicitly by Sample_ID.
    order=df["Sample_ID"].tolist()
    meta=meta.set_index("Sample_ID").loc[order].reset_index(); merged=merged.set_index("Sample_ID").loc[order].reset_index()
    df["Group_short"]=df["Group"].map(GROUP_MAP); meta["Group_short"]=meta["Group"].map(GROUP_MAP)
    protein_cols=[c for c in df.columns[26:1488] if pd.api.types.is_numeric_dtype(df[c]) and (pd.to_numeric(df[c],errors="coerce")>0).any()]
    metab_cols=[c for c in df.columns[1488:] if c!="Group_short" and pd.api.types.is_numeric_dtype(df[c]) and (pd.to_numeric(df[c],errors="coerce")>0).any()]
    # Age and sex are explicit sensitivity variables; they are never primary predictors.
    clinical_cols=[c for c in df.columns[4:26] if pd.api.types.is_numeric_dtype(df[c])]
    import json
    with open(Path(a.results)/"analysis_summary.json") as h: summary=json.load(h)
    generate_figure1(df,meta,clinical_cols,protein_cols,metab_cols,outdir)
    generate_figure2(df,meta,protein_cols,metab_cols,outdir)
    generate_figure3(df,meta,merged,summary,protein_cols,metab_cols,outdir)
    generate_figure4(df,meta,merged,Path(a.results),protein_cols,metab_cols,outdir)
    generate_figure5(outdir)
    pd.DataFrame({"Layer":["Clinical","Proteome","Metabolome"],"Variables":[len(clinical_cols),len(protein_cols),len(metab_cols)]}).to_csv(outdir/"supplementary_figure_feature_counts.csv",index=False)
    print(f"Generated Supplementary Figures 1-5 in {outdir}")


if __name__ == "__main__":
    main()
