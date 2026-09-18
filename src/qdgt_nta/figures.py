"""Recreate data-supported panels from the paper's supplementary tables."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _table(book, name):
    return pd.read_excel(book, sheet_name=name, header=1)


def _save(fig, output, stem):
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=220)
    plt.close(fig)


def create_figures(workbook, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    s1 = _table(workbook, "Table S1")
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    molecules = [Chem.MolFromSmiles(value) if isinstance(value, str) else None for value in s1["SMILES"]]
    mw = [Descriptors.MolWt(mol) for mol in molecules if mol is not None]
    logp = [Descriptors.MolLogP(mol) for mol in molecules if mol is not None]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, label in zip(axes,
        (pd.to_numeric(s1["D (×10-6, cm2 s-1)"], errors="coerce").dropna(), mw, logp),
        ("Diffusion coefficient (10⁻⁶ cm² s⁻¹)", "Molecular weight (g mol⁻¹)", "logP")):
        ax.hist(values, bins=20, color="#579e8c", edgecolor="white")
        ax.set(xlabel=label, ylabel="Compounds")
    _save(fig, output, "figure_s1_distributions")

    s5 = _table(workbook, "Table S5")
    fig, ax = plt.subplots(figsize=(6, 5))
    for mode, group in s5.groupby("Mode"):
        valid = group[["logIEpre", "logRF"]].apply(pd.to_numeric, errors="coerce").dropna()
        points = ax.scatter(valid["logIEpre"], valid["logRF"], label=mode, alpha=.8)
        slope, intercept = np.polyfit(valid["logIEpre"], valid["logRF"], 1)
        xx = np.linspace(valid["logIEpre"].min(), valid["logIEpre"].max(), 100)
        ax.plot(xx, slope * xx + intercept, "--", color=points.get_facecolor()[0], label=f"{mode} fit")
    ax.set(xlabel="Predicted logIE", ylabel="Experimental logRF")
    ax.legend()
    _save(fig, output, "figure_s7_ie_rf")

    s10 = _table(workbook, "Table S10")
    d = s10["In ADD"].eq(True)
    pos = s10["In ADIE(+)"].eq(True)
    neg = s10["In ADIE(-)"].eq(True)
    counts = s10.loc[d & (pos | neg), "Classification"].value_counts().head(20)
    categories = counts.index[::-1]
    positive = s10.loc[d & pos, "Classification"].value_counts().reindex(categories, fill_value=0)
    negative = s10.loc[d & neg, "Classification"].value_counts().reindex(categories, fill_value=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 2.5]})
    ax1.bar(["D", "IE+", "IE−", "D ∩ IE+", "D ∩ IE−", "D ∩ (IE+ ∪ IE−)"],
            [d.sum(), pos.sum(), neg.sum(), (d & pos).sum(), (d & neg).sum(), (d & (pos | neg)).sum()], color="#579e8c")
    ax1.tick_params(axis="x", rotation=65)
    ax1.set_ylabel("Compounds")
    y = np.arange(len(categories))
    ax2.barh(y - .2, positive, height=.4, label="D and IE+", color="#579e8c")
    ax2.barh(y + .2, negative, height=.4, label="D and IE−", color="#777e88")
    ax2.set_yticks(y, categories)
    ax2.set_xlabel("Compounds")
    ax2.legend()
    _save(fig, output, "figure_3b_ad_coverage")

    s11 = _table(workbook, "Table S11")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, measured, predicted, unit in [
        (axes[0], "CS-exp (μg L-1)", "CS-pred (μg L-1)", "μg L⁻¹"),
        (axes[1], "CDGT-exp (ng L-1)", "CDGT-pred (ng L-1)", "ng L⁻¹"),
    ]:
        for mode, group in s11.groupby("Mode"):
            x = pd.to_numeric(group[measured], errors="coerce")
            y = pd.to_numeric(group[predicted], errors="coerce")
            ax.scatter(x, y, label=mode, alpha=.8)
        valid = s11[[measured, predicted]].apply(pd.to_numeric, errors="coerce").dropna()
        lo = max(min(valid.min()), 1e-3)
        hi = max(valid.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xscale="log", yscale="log", xlabel=f"Measured ({unit})", ylabel=f"Predicted ({unit})")
        ax.legend()
    _save(fig, output, "figure_4a_validation")

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    for ax, table, mode, color in zip(axes, ("Table S14", "Table S15"), ("ESI+", "ESI−"), ("#348f87", "#bd7358")):
        data = _table(workbook, table)
        data["Predicted CDGT (ngL-1)"] = pd.to_numeric(data["Predicted CDGT (ngL-1)"], errors="coerce")
        data = data.dropna(subset=["Name", "Predicted CDGT (ngL-1)"]).sort_values("Predicted CDGT (ngL-1)", ascending=False).head(25)
        ax.barh(data["Name"].iloc[::-1], data["Predicted CDGT (ngL-1)"].iloc[::-1], color=color)
        ax.set(xlabel="Predicted DGT concentration (ng L⁻¹)", title=f"{mode}: top 25")
    _save(fig, output, "figure_5_predicted_concentrations")

    s12 = _table(workbook, "Table S12")
    s13 = _table(workbook, "Table S13")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, column in zip(axes, ("Prediciton fold error (CS)", "Prediciton fold error (CDGT)")):
        series = []
        for table in (s11, s12, s13):
            if column in table:
                values = pd.to_numeric(table[column], errors="coerce").dropna()
            elif "CDGT" in column:
                matched = table[["Name", "Predicted CDGT (ng L-1)"]].merge(
                    s11[["Name", "CDGT-exp (ng L-1)"]], on="Name", how="inner")
                predicted = pd.to_numeric(matched["Predicted CDGT (ng L-1)"], errors="coerce")
                measured = pd.to_numeric(matched["CDGT-exp (ng L-1)"], errors="coerce")
                values = np.maximum(predicted / measured, measured / predicted).replace([np.inf, -np.inf], np.nan).dropna()
            else:
                continue
            series.append(values)
        ax.boxplot(series, labels=("This study", "Surrogate", "RF"), showfliers=False)
        ax.set(yscale="log", ylabel="Fold prediction error", title="Aqueous" if "CS" in column else "DGT")
    _save(fig, output, "figure_s8_fold_errors")
    return ["figure_s1_distributions.png", "figure_s7_ie_rf.png", "figure_3b_ad_coverage.png", "figure_4a_validation.png", "figure_5_predicted_concentrations.png", "figure_s8_fold_errors.png"]
