"""Executa todos os runs (braço × ratio × seed) e depois a análise.

Uso:
    uv run python run_experiment.py            # experimento completo
    uv run python run_experiment.py --smoke    # teste rápido do pipeline (~minutos)
    uv run python run_experiment.py --ratios 50 --seeds 42 --arms hybrid_restricted tinybert_augmented

Os resultados são gravados incrementalmente em results/runs.csv; rodar de novo
retoma de onde parou (runs já presentes são pulados).
"""

import argparse
import logging
import warnings

import pandas as pd

from config import (
    CNN_EPOCHS,
    CNN_OOF_FOLDS,
    LIVE_GENERATION,
    NUM_EPOCHS,
    RATIO_ARMS,
    RATIOS,
    REFERENCE_ARMS,
    RESULTS_DIR,
    SEEDS,
)
from src import analysis
from src.bert import PretrainedCache, resolve_device
from src.data import load_test
from src.runner import Context, Settings, run_one, warmup


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ratios", type=int, nargs="+", default=RATIOS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--arms", nargs="+", default=REFERENCE_ARMS + RATIO_ARMS)
    p.add_argument("--smoke", action="store_true", help="subamostra + 1 época, grava em results_smoke/")
    p.add_argument("--skip-analysis", action="store_true")
    return p.parse_args()


def _plan(arms: list[str], ratios: list[int], seeds: list[int]) -> list[tuple[str, int, int]]:
    plan = []
    for seed in seeds:
        for arm in arms:
            if arm in REFERENCE_ARMS:
                plan.append((arm, 0, seed))
        for ratio in ratios:
            for arm in arms:
                if arm not in REFERENCE_ARMS:
                    plan.append((arm, ratio, seed))
    return plan


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("transformers", "httpx", "huggingface_hub", "spacy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logger = logging.getLogger("run_experiment")

    if args.smoke:
        settings = Settings(bert_epochs=1, cnn_epochs=2, oof_folds=2, train_limit=200, latency_samples=20)
        results_dir = RESULTS_DIR.parent / "results_smoke"
        args.seeds, args.ratios = args.seeds[:1], args.ratios[-1:]
    else:
        settings = Settings(
            bert_epochs=NUM_EPOCHS, cnn_epochs=CNN_EPOCHS, oof_folds=CNN_OOF_FOLDS,
            live_generation=LIVE_GENERATION,
        )
        results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions").mkdir(exist_ok=True)

    test_df = load_test()
    if args.smoke:
        test_df = test_df.sample(n=100, random_state=0).reset_index(drop=True)

    runs_csv = results_dir / "runs.csv"
    rows = pd.read_csv(runs_csv).to_dict("records") if runs_csv.exists() else []
    done = {(r["arm"], int(r["ratio"]), int(r["seed"])) for r in rows}

    plan = [p for p in _plan(args.arms, args.ratios, args.seeds) if p not in done]
    logger.info("%d run(s) a executar (%d já concluídos)", len(plan), len(done))

    if plan:
        ctx = Context(settings, test_df, PretrainedCache(), results_dir / "_work")
        logger.info("Aquecendo backend (%s) …", resolve_device())
        warmup(ctx)
        for i, (arm, ratio, seed) in enumerate(plan, 1):
            logger.info("=" * 70)
            logger.info("[%d/%d] arm=%s ratio=%d%% seed=%d", i, len(plan), arm, ratio, seed)
            row, preds = run_one(arm, ratio, seed, ctx)
            logger.info(
                "  F1 macro=%.4f  acc=%.4f  treino=%.1fs  FLOPs totais=%.3g  latência=%.2fms",
                row["f1_macro"], row["accuracy"], row["train_time_s"], row["total_flops"], row["infer_latency_ms"],
            )
            preds.to_csv(results_dir / "predictions" / f"{arm}__{ratio}__{seed}.csv", index=False)
            rows.append(row)
            pd.DataFrame(rows).to_csv(runs_csv, index=False)

    if not args.skip_analysis:
        analysis.run(results_dir)


if __name__ == "__main__":
    main()
