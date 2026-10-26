"""Análise custo-benefício: estatística pareada, veredito por contraste, gráficos e relatório.

Saídas em <results_dir>/:
    summary.csv          média ± desvio por (braço, ratio)
    contrasts.csv        ΔF1 (IC 95% bootstrap pareado), Δcusto e veredito por contraste/ratio/custo
    efficiency.csv       ganho de F1 por unidade de custo extra vs. tinybert_restricted
    fig_*.png            gráficos
    report.md            relatório que responde à pergunta de pesquisa
"""

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import EngFormatter, FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from config import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    COST_METRICS,
    LABEL_NAMES,
    NON_INFERIORITY_MARGIN,
    PRIMARY_CONTRAST,
    SECONDARY_CONTRASTS,
)
from src.generation import sec_per_sentence  # noqa: E402

logger = logging.getLogger(__name__)

QUALITY = ["f1_macro", "accuracy", "f1_weighted", "precision_macro", "recall_macro"]
COSTS = [
    "total_flops", "train_flops", "generation_flops",
    "total_time_s", "train_time_s", "generation_time_s",
    "cnn_train_s", "cnn_oof_s", "bert_train_s", "train_cpu_s",
    "peak_rss_mb", "peak_accel_mb", "params_total",
    "infer_latency_ms", "infer_throughput_sps", "infer_flops_per_sample",
]
COST_LABELS = {
    "total_flops": "FLOPs totais (treino + geração LLM)",
    "total_time_s": "Tempo total (s, treino + geração LLM)",
    "train_flops": "FLOPs só de treino",
    "train_time_s": "Tempo só de treino (s)",
}
ARM_LABELS = {
    "tinybert_full": "TinyBERT (100% real)",
    "tinybert_restricted": "TinyBERT (escasso)",
    "tinybert_augmented": "TinyBERT + DA",
    "textcnn_restricted": "TextCNN (escasso)",
    "hybrid_restricted": "TextCNN→TinyBERT (TL)",
    "hybrid_augmented": "TL + DA",
}

# Paleta de referência (skill dataviz): três primeiros slots validados all-pairs
# para as séries destacadas; referências em cinza com marcador distinto.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARM_STYLE = {
    "hybrid_restricted": {"color": "#2a78d6", "marker": "o"},
    "tinybert_augmented": {"color": "#eb6834", "marker": "D"},
    "hybrid_augmented": {"color": "#1baf7a", "marker": "P"},
    "tinybert_restricted": {"color": "#8a8984", "marker": "s"},
    "textcnn_restricted": {"color": "#52514e", "marker": "^"},
}


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------

def _macro_f1_boot(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """F1 macro vetorizado para cada linha de índices bootstrap (B, n) -> (B,)."""
    t, p = y_true[idx], y_pred[idx]
    f1s = []
    for k in LABEL_NAMES.values():
        tp = ((t == k) & (p == k)).sum(1)
        fp = ((t != k) & (p == k)).sum(1)
        fn = ((t == k) & (p != k)).sum(1)
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)


def _load_preds(results_dir: Path, arm: str, ratio: int, seed: int) -> pd.DataFrame | None:
    path = results_dir / "predictions" / f"{arm}__{ratio}__{seed}.csv"
    return pd.read_csv(path) if path.exists() else None


def paired_bootstrap(
    results_dir: Path, treat: str, comp: str, ratio: int, seeds: list[int]
) -> dict | None:
    """ΔF1 macro (treat − comp) médio entre seeds, IC 95% por bootstrap pareado nas frases de teste."""
    pairs = []
    for seed in seeds:
        a, b = _load_preds(results_dir, treat, ratio, seed), _load_preds(results_dir, comp, ratio, seed)
        if a is not None and b is not None:
            pairs.append((a, b))
    if not pairs:
        return None
    n = len(pairs[0][0])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    full = np.arange(n)[None, :]
    point, boot = 0.0, np.zeros(BOOTSTRAP_ITERATIONS)
    for a, b in pairs:
        yt = a["y_true"].to_numpy()
        pa, pb = a["y_pred"].to_numpy(), b["y_pred"].to_numpy()
        point += (_macro_f1_boot(yt, pa, full) - _macro_f1_boot(yt, pb, full))[0]
        boot += _macro_f1_boot(yt, pa, idx) - _macro_f1_boot(yt, pb, idx)
    point /= len(pairs)
    boot /= len(pairs)
    p_value = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {
        "n_seeds": len(pairs),
        "delta_f1": point,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_bootstrap": p_value,
    }


def verdict(ci_low: float, ci_high: float, delta_cost: float) -> str:
    better, worse = ci_low > 0, ci_high < 0
    non_inferior = ci_low > -NON_INFERIORITY_MARGIN
    cheaper = delta_cost < 0
    if better and cheaper:
        return "domina (melhor e mais barato)"
    if non_inferior and cheaper:
        return "custo-efetivo (não-inferior e mais barato)"
    if worse and not cheaper:
        return "dominado (pior e mais caro)"
    if better:
        return "trade-off (melhor, porém mais caro)"
    if cheaper and worse:
        return "trade-off (mais barato, porém inferior)"
    if cheaper:
        return "inconclusivo (mais barato; não-inferioridade não demonstrada)"
    return "não compensa (sem ganho significativo e mais caro)"


def contrasts(runs: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    means = runs.groupby(["arm", "ratio"])[COSTS].mean()
    for treat, comp in [PRIMARY_CONTRAST, *SECONDARY_CONTRASTS]:
        for ratio in sorted(runs.loc[runs["ratio"] > 0, "ratio"].unique()):
            if (treat, ratio) not in means.index or (comp, ratio) not in means.index:
                continue
            seeds = sorted(
                set(runs.query("arm == @treat and ratio == @ratio")["seed"])
                & set(runs.query("arm == @comp and ratio == @ratio")["seed"])
            )
            stats = paired_bootstrap(results_dir, treat, comp, int(ratio), seeds)
            if stats is None:
                continue
            for cost in COST_METRICS:
                ct, cc = means.loc[(treat, ratio), cost], means.loc[(comp, ratio), cost]
                d_cost = ct - cc
                d_pp = stats["delta_f1"] * 100
                rows.append(
                    {
                        "contrast": f"{treat} vs {comp}",
                        "treatment": treat,
                        "comparator": comp,
                        "primary": (treat, comp) == PRIMARY_CONTRAST,
                        "ratio": int(ratio),
                        **stats,
                        "cost_metric": cost,
                        "cost_treatment": ct,
                        "cost_comparator": cc,
                        "delta_cost": d_cost,
                        "cost_ratio": ct / cc if cc else math.nan,
                        # custo incremental por p.p. de F1 (análogo ao ICER)
                        "icer_cost_per_pp": d_cost / d_pp if d_pp else math.nan,
                        "verdict": verdict(stats["ci_low"], stats["ci_high"], d_cost)
                        if not math.isnan(d_cost)
                        else "custo não medido",
                    }
                )
    return pd.DataFrame(rows)


def overall_wilcoxon(runs: pd.DataFrame, treat: str, comp: str) -> dict:
    """Teste de Wilcoxon pareado sobre todos os (ratio, seed) comuns."""
    key = ["ratio", "seed"]
    a = runs[runs["arm"] == treat].set_index(key)["f1_macro"]
    b = runs[runs["arm"] == comp].set_index(key)["f1_macro"]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy()
    if len(diff) < 2 or np.allclose(diff, 0):
        return {"n_pairs": len(diff), "median_delta": float(np.median(diff)) if len(diff) else math.nan, "p_wilcoxon": math.nan}
    return {"n_pairs": len(diff), "median_delta": float(np.median(diff)), "p_wilcoxon": float(wilcoxon(diff).pvalue)}


def efficiency(runs: pd.DataFrame) -> pd.DataFrame:
    """Ganho de F1 (p.p.) por unidade de custo extra em relação ao TinyBERT escasso."""
    means = runs.groupby(["arm", "ratio"])[["f1_macro", "total_flops", "total_time_s"]].mean()
    rows = []
    for (arm, ratio), r in means.iterrows():
        if ratio == 0 or arm == "tinybert_restricted" or ("tinybert_restricted", ratio) not in means.index:
            continue
        base = means.loc[("tinybert_restricted", ratio)]
        gain_pp = (r["f1_macro"] - base["f1_macro"]) * 100
        extra_pflop = (r["total_flops"] - base["total_flops"]) / 1e15
        extra_min = (r["total_time_s"] - base["total_time_s"]) / 60
        rows.append(
            {
                "arm": arm,
                "ratio": ratio,
                "f1_gain_pp": gain_pp,
                "extra_pflop": extra_pflop,
                "extra_minutes": extra_min,
                "pp_per_pflop": gain_pp / extra_pflop if extra_pflop > 0 else math.nan,
                "pp_per_minute": gain_pp / extra_min if extra_min > 0 else math.nan,
                "dominates_reference_flops": gain_pp >= 0 and extra_pflop <= 0,
            }
        )
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in QUALITY + COSTS if c in runs]
    agg = runs.groupby(["arm", "ratio"])[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", runs.groupby(["arm", "ratio"]).size())
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def _figure(n: int, width: float = 4.2, height: float = 3.8, sharey: bool = True, max_cols: int = 4):
    """n painéis em grade de até `max_cols` colunas; devolve os eixos achatados (só os usados)."""
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(width * ncols, 9.0), height * nrows), squeeze=False, sharey=sharey
    )
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        _style(ax)
    return fig, axes[:n]


def _first_column(axes) -> list[plt.Axes]:
    return [ax for ax in axes if ax.get_subplotspec().is_first_col()]


def _pareto(points: pd.DataFrame, cost: str) -> pd.DataFrame:
    """Pontos não-dominados (menor custo, maior F1)."""
    front, best = [], -np.inf
    for _, r in points.sort_values([cost, "f1_macro"], ascending=[True, False]).iterrows():
        if r["f1_macro"] > best:
            front.append(r)
            best = r["f1_macro"]
    return pd.DataFrame(front)


def _log_cost_axis(axis, cost: str, time_unit: str = "s") -> None:
    """Ticks legíveis em eixo log de custo (1-3-10): 30 TFLOP, 100 TFLOP… ou 30 s, 100 s…"""
    flops = cost.endswith("flops")
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    axis.set_major_formatter(EngFormatter(unit="FLOP", sep=" ") if flops
                             else FuncFormatter(lambda v, _pos: f"{v:g} {time_unit}"))
    axis.set_minor_formatter(NullFormatter())


def plot_f1_vs_cost(runs: pd.DataFrame, cost: str, out: Path) -> None:
    ratios = sorted(runs.loc[runs["ratio"] > 0, "ratio"].unique())
    stats = runs.groupby(["arm", "ratio"]).agg(
        f1_macro=("f1_macro", "mean"), f1_std=("f1_macro", "std"), **{cost: (cost, "mean")}
    ).reset_index()
    if stats[cost].isna().all():
        return
    full = stats[stats["arm"] == "tinybert_full"]
    # y independente por painel: a comparação aqui é dentro do cenário (entre cenários: fig_f1_by_ratio)
    fig, axes = _figure(len(ratios), width=5.4, height=4.2, sharey=False, max_cols=2)
    for ax, ratio in zip(axes, ratios):
        pts = stats[(stats["ratio"] == ratio) & stats["arm"].isin(ARM_STYLE)].dropna(subset=[cost])
        front = _pareto(pts, cost)
        if len(front) > 1:
            ax.plot(front[cost], front["f1_macro"], color=INK_2, linestyle=":", linewidth=1.2, zorder=1)
        for _, r in pts.iterrows():
            st = ARM_STYLE[r["arm"]]
            ax.errorbar(
                r[cost], r["f1_macro"], yerr=r["f1_std"] if not math.isnan(r["f1_std"]) else None,
                color=st["color"], marker=st["marker"], markersize=8, linewidth=1.5, capsize=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARM_LABELS[r["arm"]], zorder=3,
            )
        if not full.empty and not math.isnan(full["f1_macro"].iloc[0]):
            ax.axhline(full["f1_macro"].iloc[0], color=INK_2, linestyle="--", linewidth=1)
            ax.annotate("TinyBERT 100% real", (0.02, full["f1_macro"].iloc[0]), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 4), fontsize=8, color=INK_2)
        ax.set_xscale("log")
        _log_cost_axis(ax.xaxis, cost)
        ax.set_title(f"{ratio}% dos dados reais removidos", fontsize=10, color=INK, loc="left")
        ax.set_xlabel(COST_LABELS.get(cost, cost) + " — escala log", fontsize=9, color=INK_2)
    for ax in _first_column(axes):
        ax.set_ylabel("F1 macro (média ± dp entre seeds)", fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_h = 0.5 / fig.get_figheight()  # faixa da legenda (0,5 pol.) independente da nº de linhas
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Performance × custo computacional (pontilhado = fronteira de Pareto)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, legend_h, 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_f1_by_ratio(runs: pd.DataFrame, out: Path) -> None:
    stats = runs.groupby(["arm", "ratio"])["f1_macro"].agg(["mean", "std"]).reset_index()
    fig, (ax,) = _figure(1, width=6.5, height=4.2)
    for arm, st in ARM_STYLE.items():
        s = stats[(stats["arm"] == arm)].sort_values("ratio")
        if s.empty:
            continue
        ax.plot(s["ratio"], s["mean"], color=st["color"], marker=st["marker"], markersize=8, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARM_LABELS[arm])
        ax.fill_between(s["ratio"], s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                        color=st["color"], alpha=0.12, linewidth=0)
    full = stats[stats["arm"] == "tinybert_full"]
    if not full.empty:
        ax.axhline(full["mean"].iloc[0], color=INK_2, linestyle="--", linewidth=1)
        ax.annotate("TinyBERT 100% real", (0.01, full["mean"].iloc[0]), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 4), fontsize=8, color=INK_2)
    ax.set_xticks(sorted(stats.loc[stats["ratio"] > 0, "ratio"].unique()))
    ax.set_xlabel("% dos dados reais removidos (escassez)", fontsize=9, color=INK_2)
    ax.set_ylabel("F1 macro", fontsize=9, color=INK_2)
    ax.set_title("F1 macro por nível de escassez (faixa = ± dp entre seeds)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_cost_breakdown(runs: pd.DataFrame, out: Path) -> None:
    """Barras 100% empilhadas: participação de cada etapa nos FLOPs totais, total anotado.

    A geração com LLM costuma ser ordens de grandeza maior que o treino; numa
    escala absoluta as demais etapas ficariam invisíveis.
    """
    comps = [
        ("bert_train_flops", "Fine-tuning TinyBERT", "#2a78d6"),
        ("cnn_train_flops", "Treino TextCNN (inclui cross-fitting)", "#1baf7a"),
        ("generation_flops", "Geração sintética (LLM)", "#eb6834"),
    ]
    ratios = sorted(runs.loc[runs["ratio"] > 0, "ratio"].unique())
    means = runs.groupby(["arm", "ratio"])[[c for c, *_ in comps] + ["total_flops"]].mean()
    arms = [a for a in ARM_STYLE if a in runs["arm"].unique()]
    fig, axes = _figure(len(ratios), width=5.4, height=3.0, max_cols=2)
    for ax, ratio in zip(axes, ratios):
        present = [a for a in arms if (a, ratio) in means.index]
        totals = np.array([means.loc[(a, ratio), "total_flops"] for a in present])
        left = np.zeros(len(present))
        for col, label, color in comps:
            share = np.array([means.loc[(a, ratio), col] for a in present]) / totals * 100
            ax.barh(range(len(present)), share, left=left, color=color, label=label, height=0.6,
                    edgecolor=SURFACE, linewidth=2)
            left += share
        for y, total in enumerate(totals):
            text = f"{total / 1e12:,.1f} TFLOP" if total >= 1e12 else f"{total / 1e9:,.1f} GFLOP"
            ax.annotate(text, (100, y), textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(present)), [ARM_LABELS[a] for a in present], fontsize=8, color=INK)
        ax.yaxis.set_inverted(True)  # idempotente: invert_yaxis() alternaria nos eixos compartilhados
        ax.set_xlim(0, 135)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"{ratio}% removidos", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("% dos FLOPs totais", fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0))
    fig.suptitle("Composição do custo computacional (FLOPs estimados; total à direita)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.45 / fig.get_figheight(), 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_forest(con: pd.DataFrame, out: Path) -> None:
    d = con[con["cost_metric"] == COST_METRICS[0]].copy()
    if d.empty:
        return
    d["label"] = d.apply(
        lambda r: f"{ARM_LABELS[r['treatment']]} vs {ARM_LABELS[r['comparator']]} — {r['ratio']}%", axis=1
    )
    fig, (ax,) = _figure(1, width=8, height=0.42 * len(d) + 1.4)
    y = np.arange(len(d))
    colors = ["#2a78d6" if p else "#8a8984" for p in d["primary"]]
    ax.hlines(y, d["ci_low"] * 100, d["ci_high"] * 100, color=colors, linewidth=2)
    ax.scatter(d["delta_f1"] * 100, y, color=colors, s=64, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(-NON_INFERIORITY_MARGIN * 100, color=INK_2, linewidth=1, linestyle=":")
    ax.set_yticks(y, d["label"], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("ΔF1 macro (p.p.) com IC 95% bootstrap — pontilhado = margem de não-inferioridade",
                  fontsize=9, color=INK_2)
    ax.set_title("Diferenças pareadas de F1 (azul = contraste principal)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _times_formatter(v: float, _pos) -> str:
    return f"{v:g}×"


def plot_ce_plane(con: pd.DataFrame, out: Path) -> None:
    """Plano custo-efetividade do contraste principal: ΔF1 (IC 95%) × razão de custo TL/DA.

    Um ponto por nível de escassez, ligados em ordem crescente de escassez.
    Esquerda da linha 1× = TL mais barato; acima de 0 = TL melhor.
    """
    metrics = [m for m in ("total_time_s", "total_flops") if m in set(con["cost_metric"])]
    prim = con[con["primary"] & con["cost_metric"].isin(metrics)]
    if prim.empty:
        return
    color = ARM_STYLE[PRIMARY_CONTRAST[0]]["color"]
    fig, axes = _figure(len(metrics), width=5.4, height=4.6)
    for ax, metric in zip(axes, metrics):
        d = prim[prim["cost_metric"] == metric].sort_values("ratio")
        x, y = d["cost_ratio"].to_numpy(), d["delta_f1"].to_numpy() * 100
        lo, hi = d["ci_low"].to_numpy() * 100, d["ci_high"].to_numpy() * 100
        span = max(1.0, np.abs(np.log10(x)).max()) * 1.25
        ax.set_xscale("log")
        ax.set_xlim(10**-span, 10**span)
        ax.xaxis.set_major_formatter(FuncFormatter(_times_formatter))
        ax.axvline(1, color=INK_2, linewidth=1)
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.axhline(-NON_INFERIORITY_MARGIN * 100, color=INK_2, linewidth=1, linestyle=":")
        ax.plot(x, y, color=color, linewidth=1, alpha=0.45, zorder=2)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", color=color, markersize=8, linewidth=1.5,
                    capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        for i, (xi, yi, ratio) in enumerate(zip(x, y, d["ratio"])):
            # alterna acima/abaixo para que cenários vizinhos não colidam
            ax.annotate(f"{ratio}%", (xi, yi), textcoords="offset points", xytext=(6, 5 if i % 2 else -12),
                        fontsize=8, color=INK)
        for (qx, qy, ha, va, text) in [
            (0.02, 0.98, "left", "top", "TL melhor e mais barato"),
            (0.98, 0.98, "right", "top", "TL melhor, porém mais caro"),
            (0.02, 0.02, "left", "bottom", "TL mais barato, porém pior"),
            (0.98, 0.02, "right", "bottom", "TL pior e mais caro"),
        ]:
            ax.text(qx, qy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8, color=INK_2)
        ax.set_title(COST_LABELS[metric], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("custo TL ÷ custo DA (log; < 1× = TL mais barato)", fontsize=9, color=INK_2)
    ymin = min(prim["ci_low"].min() * 100, -NON_INFERIORITY_MARGIN * 100)
    ymax = max(prim["ci_high"].max() * 100, 0)
    pad = (ymax - ymin) * 0.18
    axes[0].set_ylim(ymin - pad, ymax + pad)
    axes[0].set_ylabel("ΔF1 macro TL − DA (p.p., IC 95% bootstrap)", fontsize=9, color=INK_2)
    fig.suptitle("Plano custo-efetividade: TextCNN→TinyBERT (TL) vs TinyBERT + DA",
                 x=0.01, y=0.98, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.915, "um ponto por % de dados reais removidos · barras = IC 95% · "
             f"pontilhado = margem de não-inferioridade (−{NON_INFERIORITY_MARGIN * 100:.0f} p.p.)",
             ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _end_labels(ax: plt.Axes, ends: list[tuple[float, float, str]], min_gap_pt: float = 11) -> None:
    """Rótulos de fim de linha afastados verticalmente o mínimo para não colidirem."""
    ax.get_ylim()  # força o autoscale pendente antes de converter para pontos
    to_pt = 72 / ax.figure.dpi
    ys = [ax.transData.transform((x, y))[1] * to_pt for x, y, _ in ends]
    placed: list[float] = []
    for i in np.argsort(ys):
        placed.append(max(ys[i], placed[-1] + min_gap_pt) if placed else ys[i])
        x, y, text = ends[i]
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(8, placed[-1] - ys[i]),
                    va="center", fontsize=8, color=INK_2)


def plot_cost_by_ratio(runs: pd.DataFrame, out: Path) -> None:
    """Custo médio por nível de escassez; um painel por métrica (nunca eixo duplo)."""
    panels = [
        ("total_time_s", 1 / 60, "Tempo total (min, treino + geração LLM) — log"),
        ("total_flops", 1.0, "FLOPs totais (treino + geração LLM) — log"),
    ]
    stats = runs[runs["ratio"] > 0].groupby(["arm", "ratio"])[[c for c, *_ in panels]].mean().reset_index()
    arms = [a for a in ARM_STYLE if a in set(stats["arm"])]
    ratios = sorted(stats["ratio"].unique())
    fig, axes = _figure(len(panels), width=5.4, height=4.2, sharey=False)
    for ax, (col, scale, title) in zip(axes, panels):
        ends = []
        for arm in arms:
            st = ARM_STYLE[arm]
            s = stats[stats["arm"] == arm].sort_values("ratio")
            vals = s[col] * scale
            ax.plot(s["ratio"], vals, color=st["color"], marker=st["marker"], markersize=8, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARM_LABELS[arm])
            last = vals.iloc[-1]
            text = f"{last:.1f} min" if col == "total_time_s" else f"{last / 1e12:,.0f} TFLOP"
            ends.append((s["ratio"].iloc[-1], last, text))
        ax.set_yscale("log")
        _log_cost_axis(ax.yaxis, col, time_unit="min")
        ax.set_xticks(ratios)
        ax.set_xlim(min(ratios) - 5, max(ratios) + 14)
        _end_labels(ax, ends)
        ax.set_xlabel("% dos dados reais removidos (escassez)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(arms), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Custo computacional médio por nível de escassez (rótulo = valor no cenário mais escasso)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_delta_by_ratio(con: pd.DataFrame, out: Path) -> None:
    """ΔF1 (IC 95%) ao longo da escassez: ganho de cada estratégia e confronto direto TL − DA."""
    d = con[con["cost_metric"] == COST_METRICS[0]]
    if d.empty:
        return
    treat, comp = PRIMARY_CONTRAST
    panels = [
        ("Ganho sobre o TinyBERT escasso (só dados reais)",
         [(t, c) for t, c in SECONDARY_CONTRASTS if c == "tinybert_restricted"]),
        (f"Confronto direto: {ARM_LABELS[treat]} − {ARM_LABELS[comp]}", [PRIMARY_CONTRAST]),
    ]
    fig, axes = _figure(len(panels), width=5.4, height=4.2, sharey=False)
    for ax, (title, pairs) in zip(axes, panels):
        for t, c in pairs:
            s = d[(d["treatment"] == t) & (d["comparator"] == c)].sort_values("ratio")
            if s.empty:
                continue
            color = INK if (t, c) == PRIMARY_CONTRAST else ARM_STYLE[t]["color"]
            marker = "o" if (t, c) == PRIMARY_CONTRAST else ARM_STYLE[t]["marker"]
            label = ARM_LABELS[t] if (t, c) != PRIMARY_CONTRAST else "TL − DA"
            ax.fill_between(s["ratio"], s["ci_low"] * 100, s["ci_high"] * 100, color=color, alpha=0.12,
                            linewidth=0)
            ax.plot(s["ratio"], s["delta_f1"] * 100, color=color, marker=marker, markersize=8, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.axhline(0, color=INK_2, linewidth=1)
        if pairs == [PRIMARY_CONTRAST]:
            ax.axhline(-NON_INFERIORITY_MARGIN * 100, color=INK_2, linewidth=1, linestyle=":")
            ax.annotate("margem de não-inferioridade", (0.01, -NON_INFERIORITY_MARGIN * 100),
                        xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, -11),
                        fontsize=8, color=INK_2)
        ax.set_xticks(sorted(d["ratio"].unique()))
        ax.set_xlabel("% dos dados reais removidos (escassez)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    axes[0].set_ylabel("ΔF1 macro (p.p.; faixa = IC 95% bootstrap pareado)", fontsize=9, color=INK_2)
    fig.suptitle("Quanto cada estratégia recupera e quem vence em cada nível de escassez",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "—"
            return f"{v:.3e}" if abs(v) >= 1e5 or (0 < abs(v) < 1e-3) else f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(runs: pd.DataFrame, con: pd.DataFrame, eff: pd.DataFrame, results_dir: Path) -> None:
    treat, comp = PRIMARY_CONTRAST
    lines = [
        "# Transfer learning (TextCNN → TinyBERT) vs TinyBERT + data augmentation",
        "",
        "**Pergunta:** transfer learning oferece custo-benefício (performance / custo computacional) "
        "para classificação de texto financeiro com dados escassos?",
        "",
        f"Contraste principal: `{treat}` (TL, só dados reais) vs `{comp}` (TinyBERT + DA com {runs['n_synthetic'].max()} "
        "frases sintéticas no máximo). IC 95% por bootstrap pareado nas frases de teste, média entre seeds. "
        f"Margem de não-inferioridade: {NON_INFERIORITY_MARGIN * 100:.1f} p.p. de F1 macro.",
        "",
        "## Veredito por nível de escassez",
        "",
    ]
    prim = con[con["primary"]]
    if prim.empty:
        lines.append("_Sem dados suficientes para o contraste principal._")
    else:
        show = prim[["ratio", "cost_metric", "delta_f1", "ci_low", "ci_high", "p_bootstrap",
                     "cost_ratio", "icer_cost_per_pp", "verdict"]].copy()
        for c in ("delta_f1", "ci_low", "ci_high"):
            show[c] = show[c] * 100
        show = show.rename(columns={"delta_f1": "ΔF1 (p.p.)", "ci_low": "IC inf", "ci_high": "IC sup",
                                    "cost_ratio": "custo TL / custo DA", "icer_cost_per_pp": "Δcusto por p.p."})
        lines += [_md_table(show), ""]
        wx = overall_wilcoxon(runs, treat, comp)
        lines += [
            f"Wilcoxon pareado sobre todos os (ratio, seed): n={wx['n_pairs']}, "
            f"mediana ΔF1={wx['median_delta'] * 100:.2f} p.p., p={wx['p_wilcoxon']:.4g}",
            "",
        ]

    lines += ["## Contrastes secundários", ""]
    sec = con[~con["primary"] & (con["cost_metric"] == COST_METRICS[0])]
    if not sec.empty:
        show = sec[["contrast", "ratio", "delta_f1", "ci_low", "ci_high", "cost_ratio", "verdict"]].copy()
        for c in ("delta_f1", "ci_low", "ci_high"):
            show[c] = show[c] * 100
        lines += [_md_table(show), ""]

    lines += ["## Eficiência vs TinyBERT escasso (ganho de F1 por custo extra)", ""]
    if not eff.empty:
        lines += [_md_table(eff), ""]

    summary = runs.groupby(["arm", "ratio"]).agg(
        n=("seed", "count"),
        f1_macro=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        accuracy=("accuracy", "mean"),
        generation_time_s=("generation_time_s", "mean"),
        train_time_s=("train_time_s", "mean"),
        total_time_s=("total_time_s", "mean"),
        total_flops=("total_flops", "mean"),
        latency_ms=("infer_latency_ms", "mean"),
        params=("params_total", "mean"),
    ).reset_index()
    lines += ["## Resumo por braço", "", _md_table(summary), ""]

    if "selected_features" in runs:
        sel = runs.dropna(subset=["selected_features"])
        if not sel.empty:
            counts = (
                sel["selected_features"].str.split(";").explode().value_counts().rename_axis("feature")
                .reset_index(name="n_seeds_selecionada")
            )
            lines += [
                f"## Features da TextCNN usadas pelo TinyBERT (top-{int(sel['n_cnn_features'].iloc[0])} por informação mútua)",
                "",
                "Por run (ordem decrescente de importância):",
                "",
                *[f"- `{r.arm}` {r.ratio}% seed {r.seed}: {r.selected_features.replace(';', ', ')}"
                  for r in sel.itertuples()],
                "",
                "Frequência entre runs:",
                "",
                _md_table(counts),
                "",
            ]
    lines += [
        "## Figuras",
        "",
        "- `fig_f1_vs_cost_<custo>.png` — performance × custo com fronteira de Pareto "
        "(total_* inclui a geração LLM; train_* considera só o treino; total_time_s requer tempo de geração medido)",
        "- `fig_f1_by_ratio.png` — F1 por nível de escassez",
        "- `fig_cost_breakdown.png` — composição do custo",
        "- `fig_delta_f1_forest.png` — ΔF1 pareado com IC 95%",
        "- `fig_ce_plane.png` — plano custo-efetividade TL vs DA (ΔF1 × razão de custo), um ponto por escassez",
        "- `fig_cost_by_ratio.png` — custo total (tempo e FLOPs) de cada braço ao longo da escassez",
        "- `fig_delta_f1_by_ratio.png` — ganho de TL e DA sobre o TinyBERT escasso e confronto TL − DA por escassez",
        "",
    ]
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(results_dir: Path) -> None:
    runs_csv = results_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"{runs_csv} não existe — rode run_experiment.py primeiro.")
    runs = pd.read_csv(runs_csv)
    logger.info("Analisando %d runs de %s", len(runs), runs_csv)

    # Runs sem tempo de geração medido (CSVs baixados) podem usar a medição
    # posterior de measure_generation_cost.py; geração ao vivo nunca é sobrescrita.
    sps = sec_per_sentence()
    missing = runs["generation_time_s"].isna()
    if sps is not None and missing.any():
        runs.loc[missing, "generation_time_s"] = runs.loc[missing, "n_synthetic"] * sps
        runs["total_time_s"] = runs["train_time_s"] + runs["generation_time_s"]
        logger.info("Tempo de geração estimado (%.2fs/frase) para %d run(s)", sps, missing.sum())

    summarize(runs).to_csv(results_dir / "summary.csv", index=False)
    con = contrasts(runs, results_dir)
    con.to_csv(results_dir / "contrasts.csv", index=False)
    eff = efficiency(runs)
    eff.to_csv(results_dir / "efficiency.csv", index=False)

    for cost in COST_METRICS:
        plot_f1_vs_cost(runs, cost, results_dir / f"fig_f1_vs_cost_{cost}.png")
    plot_f1_by_ratio(runs, results_dir / "fig_f1_by_ratio.png")
    plot_cost_breakdown(runs, results_dir / "fig_cost_breakdown.png")
    plot_forest(con, results_dir / "fig_delta_f1_forest.png")
    plot_ce_plane(con, results_dir / "fig_ce_plane.png")
    plot_cost_by_ratio(runs, results_dir / "fig_cost_by_ratio.png")
    plot_delta_by_ratio(con, results_dir / "fig_delta_f1_by_ratio.png")
    write_report(runs, con, eff, results_dir)
    logger.info("Relatório → %s", results_dir / "report.md")
