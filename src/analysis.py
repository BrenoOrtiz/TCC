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
    HUMAN_ANNOTATION_SEC_PER_SAMPLE,
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


def _log_cost_axis(axis, cost: str, time_unit: str = "s") -> None:
    """Ticks legíveis em eixo log de custo (1-3-10): 30 TFLOP, 100 TFLOP… ou 30 s, 100 s…"""
    flops = cost.endswith("flops")
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    axis.set_major_formatter(EngFormatter(unit="FLOP", sep=" ") if flops
                             else FuncFormatter(lambda v, _pos: f"{v:g} {time_unit}"))
    axis.set_minor_formatter(NullFormatter())


def _headline(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Título com a conclusão da figura + subtítulo com como ler; alturas fixas em polegadas."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=12, color=INK)
    fig.text(0.01, 1 - 0.45 / h, subtitle, ha="left", va="top", fontsize=9, color=INK_2, linespacing=1.4)


def _end_labels(ax: plt.Axes, ends: list[tuple[float, float, str]], min_gap_pt: float = 11) -> None:
    """Rótulos de fim de linha afastados verticalmente o mínimo para não colidirem."""
    ax.get_ylim()  # força o autoscale pendente antes de converter para pontos
    to_pt = 72 / ax.figure.dpi
    ys = [ax.transData.transform((x, y))[1] * to_pt for x, y, _ in ends]
    placed: list[float] = []
    for i in np.argsort(ys):
        placed.append(max(ys[i], placed[-1] + min_gap_pt) if placed else ys[i])
        x, y, text = ends[i]
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(10, placed[-1] - ys[i]),
                    va="center", fontsize=9, color=INK)


FOCUS_ARMS = ["tinybert_augmented", "hybrid_restricted", "tinybert_restricted"]  # ordem de leitura
SHORT_LABELS = {"tinybert_augmented": "DA", "hybrid_restricted": "TL", "tinybert_restricted": "escasso"}


def plot_f1_by_ratio(runs: pd.DataFrame, out: Path) -> None:
    """Qualidade ao longo da escassez: empate até 50%, colapso do TinyBERT escasso em 75%."""
    stats = runs.groupby(["arm", "ratio"])["f1_macro"].agg(["mean", "std"]).reset_index()
    ratios = sorted(stats.loc[stats["ratio"] > 0, "ratio"].unique())
    arms = [a for a in FOCUS_ARMS if a in set(stats["arm"])]
    fig, (ax,) = _figure(1, width=9, height=5.4)
    ends = []
    for k, arm in enumerate(arms):
        st, s = ARM_STYLE[arm], stats[stats["arm"] == arm].sort_values("ratio")
        x = s["ratio"] - (k - (len(arms) - 1) / 2) * 1.0  # leve deslocamento: barras de erro não se sobrepõem
        ax.errorbar(x, s["mean"], yerr=s["std"].fillna(0), color=st["color"], marker=st["marker"],
                    markersize=8, linewidth=2, capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=ARM_LABELS[arm], zorder=3)
        ends.append((x.iloc[-1], s["mean"].iloc[-1], f"{SHORT_LABELS[arm]}  {s['mean'].iloc[-1]:.3f}"))
    full = stats[stats["arm"] == "tinybert_full"]
    if not full.empty:
        f1_full = full["mean"].iloc[0]
        ax.axhline(f1_full, color=INK_2, linestyle="--", linewidth=1, zorder=1)
        ax.annotate(f"TinyBERT com 100% dos dados reais: {f1_full:.3f}", (0.01, f1_full),
                    xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, 4),
                    fontsize=8.5, color=INK_2)
    ax.axvspan(min(ratios) - 4, 50 + 4, color=GRID, alpha=0.35, linewidth=0, zorder=0)
    ax.text(30, 0.80, "10–50% removidos:\ndiferenças dentro do ruído\n(IC 95% de TL − DA cruzam zero)",
            ha="center", va="center", fontsize=9, color=INK_2)
    _end_labels(ax, ends, min_gap_pt=13)
    ax.set_xticks(ratios, [f"{r}%" for r in ratios])
    ax.set_xlim(min(ratios) - 4, max(ratios) + 12)
    ax.set_ylim(0.5, 0.93)
    ax.set_xlabel("dados reais removidos do treino (escassez)", fontsize=9, color=INK_2)
    ax.set_ylabel("F1 macro (média ± dp entre 3 seeds)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    _headline(fig, "Até 50% de escassez as três abordagens têm F1 equivalente; em 75%, DA e TL evitam o colapso",
              "Com 75% removidos, o TinyBERT treinado só com os dados restantes fica instável (dp 0.16 entre seeds); "
              "o DA recupera mais F1 que o TL.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_cost_by_ratio(runs: pd.DataFrame, out: Path) -> None:
    """Custo médio por nível de escassez; um painel por métrica (nunca eixo duplo)."""
    panels = [
        ("total_time_s", 1 / 60, "Tempo total (treino + geração LLM), escala log"),
        ("total_flops", 1.0, "FLOPs totais (treino + geração LLM), escala log"),
    ]
    stats = runs[runs["ratio"] > 0].groupby(["arm", "ratio"])[[c for c, *_ in panels]].mean()
    arms = [a for a in FOCUS_ARMS if a in stats.index.get_level_values("arm")]
    ratios = sorted(stats.index.get_level_values("ratio").unique())
    fig, axes = _figure(len(panels), width=5.6, height=5.0, sharey=False)
    for ax, (col, scale, title) in zip(axes, panels):
        ends = []
        for arm in arms:
            st = ARM_STYLE[arm]
            vals = stats.loc[arm, col].reindex(ratios) * scale
            ax.plot(ratios, vals, color=st["color"], marker=st["marker"], markersize=8, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARM_LABELS[arm])
            last = vals.iloc[-1]
            text = f"{last:.1f} min" if col == "total_time_s" else EngFormatter(unit="FLOP", sep=" ", places=0)(last)
            ends.append((ratios[-1], last, f"{SHORT_LABELS[arm]}  {text}"))
        ax.set_yscale("log")
        _log_cost_axis(ax.yaxis, col, time_unit="min")
        ax.set_xticks(ratios, [f"{r}%" for r in ratios])
        ax.set_xlim(min(ratios) - 5, max(ratios) + 22)
        _end_labels(ax, ends, min_gap_pt=13)
        ax.set_xlabel("dados reais removidos do treino (escassez)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0))
    worst = ratios[-1]
    t_da, t_tl = stats.loc[("tinybert_augmented", worst), "total_time_s"], stats.loc[("hybrid_restricted", worst), "total_time_s"]
    _headline(fig, "O custo do DA cresce com a escassez (mais frases a gerar); o do TL cai (menos dados a treinar)",
              f"Com {worst}% removidos, o DA leva {t_da / 60:.1f} min e o TL {t_tl / 60:.1f} min "
              f"({t_da / t_tl:.0f}× menos).\nEm FLOPs a diferença é de duas a três ordens de grandeza, "
              "pois a geração com o LLM de 8B domina o custo do DA.")
    fig.tight_layout(rect=(0, 0.45 / fig.get_figheight(), 1, 1 - 1.0 / fig.get_figheight()))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_ce_plane(con: pd.DataFrame, out: Path) -> None:
    """Veredito do contraste principal: ΔF1 TL − DA (IC 95%) × quantas vezes o TL é mais rápido.

    Um ponto por nível de escassez; faixa sombreada = zona de não-inferioridade.
    """
    d = con[con["primary"] & (con["cost_metric"] == "total_time_s")].sort_values("ratio")
    if d.empty:
        return
    flops = con[con["primary"] & (con["cost_metric"] == "total_flops")]
    color = ARM_STYLE[PRIMARY_CONTRAST[0]]["color"]
    x = 1 / d["cost_ratio"].to_numpy()  # custo DA ÷ custo TL: > 1 = TL mais rápido
    y, lo, hi = (d[c].to_numpy() * 100 for c in ("delta_f1", "ci_low", "ci_high"))
    margin = -NON_INFERIORITY_MARGIN * 100

    fig, (ax,) = _figure(1, width=9, height=5.6)
    ax.set_xscale("log")
    ax.set_xlim(0.35, x.max() * 2.2)
    ax.set_ylim(min(lo.min(), margin) - 1.2, max(hi.max(), 0) + 1.6)
    ax.axhspan(margin, ax.get_ylim()[1], color=GRID, alpha=0.45, linewidth=0, zorder=0)
    ax.text(0.99, 0.97, f"zona de não-inferioridade (ΔF1 ≥ −{-margin:g} p.p.)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.axhline(margin, color=INK_2, linewidth=1, linestyle=":")
    ax.axvline(1, color=INK_2, linewidth=1)
    ax.annotate("mesmo custo", (1, 0.02), xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(4, 0), fontsize=8.5, color=INK_2)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", color=color, markersize=9, linewidth=1.8,
                capsize=4, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for xi, yi, ratio, verdict_text in zip(x, y, d["ratio"], d["verdict"]):
        short = verdict_text.split(" (")[0]
        left = xi < 1  # à esquerda da linha "mesmo custo": rótulo do lado esquerdo para não cruzá-la
        ax.annotate(f"{ratio}% removidos\n{short}", (xi, yi), textcoords="offset points",
                    xytext=(-10 if left else 10, 0), ha="right" if left else "left", va="center", fontsize=9, color=INK, fontweight="bold" if short == "custo-efetivo" else "normal")
    ticks = [t for t in (1, 2, 5, 10, 20, 50) if t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("quantas vezes o TL é mais rápido que o DA (tempo total, escala log)", fontsize=9, color=INK_2)
    ax.set_ylabel("ΔF1 macro TL − DA (p.p., IC 95% bootstrap pareado)", fontsize=9, color=INK_2)
    fx = 1 / flops["cost_ratio"]
    _headline(fig, "TL × DA: F1 equivalente por uma fração do tempo até 50% de escassez; em 75%, o DA vence por 5 p.p.",
              "Um ponto por cenário; acima da linha pontilhada, o TL não é mais que 1 p.p. pior que o DA. "
              f"Em FLOPs o TL é de {fx.min():.0f}× a {fx.max():.0f}× mais barato.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_time_f1_human(runs: pd.DataFrame, out: Path) -> None:
    """Tempo total e F1 lado a lado, uma linha por método, agrupadas por cenário.

    Anotação humana (hipotética): tempo para rotular à mão o dataset existente do cenário
    (n títulos reais restantes × s/título) + treino do TinyBERT nesse dataset; F1 não definido —
    exibido só como resultado ótimo. Se houver runs de `tinybert_full`, um grupo de baseline
    (100% dos dados reais) abre a figura, com a anotação humana do dataset inteiro.
    """
    stats = runs.groupby(["arm", "ratio"]).agg(
        f1=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        time_s=("total_time_s", "mean"), train_s=("train_time_s", "mean"),
        n_train=("n_train", "mean"),
    )
    arms = stats.index.get_level_values("arm")
    order = [a for a in ("tinybert_restricted", "hybrid_restricted", "tinybert_augmented") if a in arms]
    ratios = sorted(stats.loc[stats.index.get_level_values("ratio") > 0].index.get_level_values("ratio").unique())

    def arm_row(arm, ratio, color):
        r = stats.loc[(arm, ratio)]
        return (ARM_LABELS[arm], color, r["time_s"] / 60, r["f1"], r["f1_std"], False)

    def human_row(dataset):  # dataset = linha de stats do braço treinado só no dataset anotado
        n = int(round(dataset["n_train"]))
        minutes = (n * HUMAN_ANNOTATION_SEC_PER_SAMPLE + dataset["train_s"]) / 60
        return (f"Anotação humana ({n} títulos)", INK, minutes, np.nan, np.nan, True)

    groups = []  # (cabeçalho, linhas); linha = (rótulo, cor, minutos, f1, f1_std, referência humana)
    if "tinybert_full" in arms:
        groups.append(("Baseline: 100% dos dados reais (nada removido)",
                       [arm_row("tinybert_full", 0, INK_2), human_row(stats.loc[("tinybert_full", 0)])]))
    for ratio in ratios:  # dataset existente = só os dados reais restantes (braço escasso)
        groups.append((f"Cenário: {ratio}% dos dados reais removidos",
                       [arm_row(arm, ratio, ARM_STYLE[arm]["color"]) for arm in order]
                       + [human_row(stats.loc[("tinybert_restricted", ratio)])]))

    rows, starts, y = [], [], 0
    for _, g in groups:  # uma linha vazia acima de cada grupo para o cabeçalho
        starts.append(y)
        rows += [(y + j, *r) for j, r in enumerate(g)]
        y += len(g) + 2
    d = pd.DataFrame(rows, columns=["y", "label", "color", "minutes", "f1", "f1_std", "assumed"])

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 0.4 * (d["y"].max() + 2.6)), sharey=True,
                                     gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.05})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_t, ax_f):
        _style(ax)
        ax.grid(False, axis="y")
        for i, ((_, g), start) in enumerate(zip(groups, starts)):  # faixa alternada separa os cenários
            if i % 2 == 0:
                ax.axhspan(start - 1.6, start + len(g) - 0.4, color=GRID, alpha=0.45, linewidth=0, zorder=0)

    # Tempo total: barras lineares com o valor escrito
    t_max = d["minutes"].max()
    ax_t.barh(d["y"], d["minutes"], color=d["color"], height=0.7, zorder=2)
    for _, r in d.iterrows():
        txt = f"{r['minutes']:.1f} min" if r["minutes"] < 10 else f"{r['minutes']:.0f} min"
        ax_t.text(r["minutes"] + t_max * 0.01, r["y"], txt, va="center", fontsize=8.5,
                  color=INK, fontweight="bold" if r["assumed"] else "normal")
    ax_t.set_xlim(0, t_max * 1.15)
    ax_t.set_xlabel("Tempo total (min) — treino + geração LLM, ou anotação + treino", fontsize=9, color=INK_2)
    ax_t.set_title("Tempo total", fontsize=10, color=INK, loc="left")

    # F1: ponto ± dp com o valor escrito
    # F1: barras com eixo recortado (todos os valores ficam acima de x_lo) e valor escrito
    x_lo = np.floor((d["f1"].min() - 0.05) * 10) / 10
    x_hi = 1.0
    for _, r in d.iterrows():
        if r["assumed"]:  # anotação humana: sem F1 numérico, só a referência de resultado ótimo
            ax_f.barh(r["y"], x_hi - x_lo, left=x_lo, height=0.7, zorder=2, color=SURFACE,
                      edgecolor=INK, hatch="///", linewidth=1.2)
            ax_f.text((x_lo + x_hi) / 2, r["y"], "resultado ótimo (referência)", ha="center", va="center",
                      fontsize=8.5, color=INK, fontweight="bold", zorder=4,
                      bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})
            continue
        ax_f.barh(r["y"], r["f1"] - x_lo, left=x_lo, height=0.7, zorder=2, color=r["color"])
        if not np.isnan(r["f1_std"]):
            ax_f.errorbar(r["f1"], r["y"], xerr=r["f1_std"], color=INK_2, linewidth=1, capsize=2.5,
                          fmt="none", zorder=3)
        right = r["f1"] + (0 if np.isnan(r["f1_std"]) else r["f1_std"])
        ax_f.text(right + 0.004, r["y"], f"{r['f1']:.3f}", va="center", fontsize=8.5, color=INK)
    ax_f.set_xlim(x_lo, x_hi)
    ax_f.set_xlabel(f"F1 macro (média ± dp entre seeds) — eixo começa em {x_lo:g}", fontsize=9, color=INK_2)
    ax_f.set_title("F1 macro", fontsize=10, color=INK, loc="left")

    ax_t.set_yticks(d["y"], d["label"], fontsize=9)
    for tick, color in zip(ax_t.get_yticklabels(), d["color"]):
        tick.set_color(color)
    ax_t.set_ylim(d["y"].max() + 0.8, -1.8)  # invertido, com espaço para o 1º cabeçalho
    ax_t.tick_params(axis="y", length=0)
    ax_f.tick_params(axis="y", length=0)
    for (header, _), start in zip(groups, starts):  # cabeçalho de cada grupo na linha vazia acima dele
        ax_t.text(0.005, start - 1.05, header,
                  transform=ax_t.get_yaxis_transform(), ha="left", va="center",
                  fontsize=10, color=INK, fontweight="bold")

    fig.suptitle("Tempo total × F1 por cenário de escassez: TL, DA e anotação humana",
                 x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 1 - 0.38 / fig.get_figheight(),
             f"Anotação humana = títulos do dataset existente (no baseline, o dataset inteiro) × "
             f"{HUMAN_ANNOTATION_SEC_PER_SAMPLE:g} s + treino do TinyBERT nesse dataset.\n"
             "No F1, a anotação humana aparece só como resultado ótimo de referência (sem valor definido).",
             ha="left", va="top", fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.1 / fig.get_figheight(), bottom=0.6 / fig.get_figheight())
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
        "- `fig_f1_by_ratio.png` — F1 macro de TL, DA e TinyBERT escasso por nível de escassez, "
        "com o baseline de 100% dos dados",
        "- `fig_cost_by_ratio.png` — custo total (tempo e FLOPs) de cada braço ao longo da escassez",
        "- `fig_ce_plane.png` — veredito TL vs DA: ΔF1 (IC 95%) × quantas vezes o TL é mais rápido, um ponto por cenário",
        "- `fig_time_f1_human.png` — tempo total × F1 por cenário, com baseline de 100% dos dados e "
        f"anotação humana ({HUMAN_ANNOTATION_SEC_PER_SAMPLE:g} s/título × títulos do dataset existente)",
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

    plot_f1_by_ratio(runs, results_dir / "fig_f1_by_ratio.png")
    plot_cost_by_ratio(runs, results_dir / "fig_cost_by_ratio.png")
    plot_ce_plane(con, results_dir / "fig_ce_plane.png")
    plot_time_f1_human(runs, results_dir / "fig_time_f1_human.png")
    write_report(runs, con, eff, results_dir)
    logger.info("Relatório → %s", results_dir / "report.md")
