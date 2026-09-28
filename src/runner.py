"""Executa um run (braço × ratio × seed): treino, avaliação e medição de custo."""

import gc
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import CNN_TOP_K_FEATURES, EVAL_BATCH_SIZE, LATENCY_SAMPLES, MONITOR_INTERVAL_S
from src import bert, textcnn
from src.cost import ResourceMonitor, count_params, sync_device, torch_forward_flops, training_flops
from src.data import arm_data_variant, arm_model, load_synthetic, load_train
from src.generation import generate_live, generation_cost

logger = logging.getLogger(__name__)

LATENCY_WARMUP = 5


@dataclass
class Settings:
    bert_epochs: int
    cnn_epochs: int
    oof_folds: int
    train_limit: int | None = None  # subamostra para smoke test
    live_generation: bool = False   # braço DA gera os sintéticos via Ollama (tempo real)
    latency_samples: int = LATENCY_SAMPLES


@dataclass
class Context:
    settings: Settings
    test_df: pd.DataFrame
    pretrained: bert.PretrainedCache
    work_dir: Path
    # CNN final por (variante de dados, ratio, seed) — reusada por textcnn_* e hybrid_*;
    # o custo é cobrado integralmente em cada braço, como se rodasse sozinho.
    cnn_cache: dict[tuple, textcnn.CNNTraining] = field(default_factory=dict)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _time_latency(predict_one, texts: list[str]) -> float:
    """Mediana (ms) de inferência ponta-a-ponta com batch=1."""
    for text in texts[:LATENCY_WARMUP]:
        predict_one(text)
    times = []
    for text in texts:
        sync_device()
        t0 = time.perf_counter()
        predict_one(text)
        sync_device()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _load_arm_data(arm: str, ratio: int, seed: int, settings: Settings) -> pd.DataFrame:
    variant = arm_data_variant(arm)
    train_df = load_train(variant, None if variant == "full" else ratio)
    if settings.train_limit:
        train_df = train_df.sample(
            n=min(settings.train_limit, len(train_df)), random_state=seed
        ).reset_index(drop=True)
    return train_df


def _cnn(ctx: Context, arm: str, ratio: int, seed: int, texts, labels) -> textcnn.CNNTraining:
    key = (arm_data_variant(arm), ratio, seed)
    if key not in ctx.cnn_cache:
        ctx.cnn_cache[key] = textcnn.train_textcnn(texts, labels, seed, ctx.settings.cnn_epochs)
    return ctx.cnn_cache[key]


def warmup(ctx: Context) -> None:
    """Treino/inferência descartáveis para que o 1º run medido não pague a
    inicialização do backend (compilação de kernels MPS/CUDA, alocadores, spaCy)."""
    texts = ctx.test_df["text"].tolist()[:64]
    labels = ctx.test_df["label"].tolist()[:64]
    cnn = textcnn.train_textcnn(texts, labels, seed=0, epochs=1)
    feats = textcnn.extract_features(cnn.nlp, texts)[:, :CNN_TOP_K_FEATURES]
    model = ctx.pretrained.new_model(0, CNN_TOP_K_FEATURES)
    dataset = bert.EncodedDataset(texts, labels, ctx.pretrained.tokenizer, feats)
    bert.train(model, dataset, 0, 1, ctx.work_dir / "trainer")
    bert.predict_logits(model, texts, ctx.pretrained.tokenizer, EVAL_BATCH_SIZE, feats)
    del model
    _free_memory()


def run_one(arm: str, ratio: int, seed: int, ctx: Context) -> tuple[dict, pd.DataFrame]:
    s = ctx.settings
    model_kind = arm_model(arm)
    live = None
    if arm_data_variant(arm) == "augmented" and s.live_generation:
        # DA ponta a ponta: gera r·N frases agora e junta ao conjunto restrito
        live = generate_live(load_train("full", None), ratio, seed)
        restricted = load_train("restricted", ratio).assign(source="original")
        train_df = (
            pd.concat([restricted, live.synthetic], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
    else:
        train_df = _load_arm_data(arm, ratio, seed, s)
    texts, labels = train_df["text"].tolist(), train_df["label"].tolist()
    test_texts = ctx.test_df["text"].tolist()
    n_synthetic = int((train_df.get("source") == "synthetic").sum()) if "source" in train_df else 0

    row: dict = {
        "arm": arm,
        "model": model_kind,
        "data": arm_data_variant(arm),
        "ratio": ratio,
        "seed": seed,
        "n_train": len(train_df),
        "n_synthetic": n_synthetic,
        "device": str(bert.resolve_device()) if model_kind != "textcnn" else "cpu",
        "cnn_train_s": 0.0,
        "cnn_oof_s": 0.0,
        "cnn_feat_s": 0.0,
        "bert_train_s": 0.0,
        "cnn_train_flops": 0.0,
        "bert_train_flops": 0.0,
        "generation_flops": 0.0,
        "generation_time_s": 0.0,
    }

    if live is not None:
        row["generation_time_s"] = live.seconds
        row["generation_flops"] = live.flops
        row["gen_prompt_tokens"] = live.prompt_tokens
        row["gen_output_tokens"] = live.output_tokens
        row["gen_calls"] = live.calls
    elif n_synthetic:
        synthetic = train_df[train_df["source"] == "synthetic"]
        if s.train_limit is None:
            synthetic = load_synthetic(ratio)
        row["generation_flops"], row["generation_time_s"] = generation_cost(synthetic)

    _free_memory()
    monitor = ResourceMonitor(interval=MONITOR_INTERVAL_S)
    with monitor:
        # ------------------------------------------------------------------ CNN
        cnn = None
        train_features = None
        if model_kind in {"textcnn", "hybrid"}:
            cnn = _cnn(ctx, arm, ratio, seed, texts, labels)
            n_tokens = textcnn.count_tokens(cnn.nlp, texts)
            row["cnn_train_s"] = cnn.train_s
            row["cnn_train_flops"] = training_flops(cnn.flops_per_token, n_tokens, s.cnn_epochs)

        if model_kind == "hybrid":
            if s.oof_folds >= 2:
                train_features, row["cnn_oof_s"] = textcnn.out_of_fold_features(
                    texts, labels, seed, s.oof_folds, s.cnn_epochs
                )
                # cada fold treina em (k-1)/k dos dados -> k·(k-1)/k = (k-1) passadas extras
                row["cnn_train_flops"] *= 1 + (s.oof_folds - 1)
                row["cnn_train_flops"] += cnn.flops_per_token * n_tokens  # extração held-out
            else:
                t0 = time.perf_counter()
                train_features = textcnn.extract_features(cnn.nlp, texts)
                row["cnn_feat_s"] = time.perf_counter() - t0
                row["cnn_train_flops"] += cnn.flops_per_token * n_tokens

            # feature extraction: só as K features mais informativas seguem para o TinyBERT
            t0 = time.perf_counter()
            selected = textcnn.select_top_features(train_features, labels, CNN_TOP_K_FEATURES, seed)
            train_features = np.ascontiguousarray(train_features[:, selected])
            row["cnn_feat_s"] += time.perf_counter() - t0
            row["n_cnn_features"] = len(selected)
            row["selected_features"] = ";".join(textcnn.FEATURE_NAMES[i] for i in selected)

        # -------------------------------------------------------------- TinyBERT
        model = None
        bert_fwd_flops = 0
        if model_kind in {"tinybert", "hybrid"}:
            feature_dim = train_features.shape[1] if model_kind == "hybrid" else 0
            model = ctx.pretrained.new_model(seed, feature_dim)
            bert_fwd_flops = torch_forward_flops(
                model, bert.flops_inputs(ctx.pretrained.tokenizer, feature_dim)
            )
            dataset = bert.EncodedDataset(texts, labels, ctx.pretrained.tokenizer, train_features)
            t0 = time.perf_counter()
            bert.train(model, dataset, seed, s.bert_epochs, ctx.work_dir / "trainer")
            sync_device()
            row["bert_train_s"] = time.perf_counter() - t0
            row["bert_train_flops"] = training_flops(bert_fwd_flops, len(dataset), s.bert_epochs)

    row["train_cpu_s"] = monitor.cpu_s
    row["peak_rss_mb"] = monitor.peak_rss_mb
    row["peak_accel_mb"] = monitor.peak_accel_mb

    # ------------------------------------------------------------ parâmetros
    cnn_params = cnn.n_params if cnn else 0
    bert_total, bert_trainable = count_params(model) if model else (0, 0)
    row["params_total"] = cnn_params + bert_total
    row["params_trainable"] = cnn_params + bert_trainable

    # ------------------------------------------------ avaliação + inferência
    tok = ctx.pretrained.tokenizer
    if model_kind == "textcnn":
        def predict_batch(batch: list[str]) -> np.ndarray:
            return np.log(textcnn.predict_proba(cnn.nlp, batch) + 1e-12)
    elif model_kind == "tinybert":
        def predict_batch(batch: list[str]) -> np.ndarray:
            return bert.predict_logits(model, batch, tok, EVAL_BATCH_SIZE)
    else:
        def predict_batch(batch: list[str]) -> np.ndarray:
            feats = np.ascontiguousarray(textcnn.extract_features(cnn.nlp, batch)[:, selected])
            return bert.predict_logits(model, batch, tok, EVAL_BATCH_SIZE, feats)

    sync_device()
    t0 = time.perf_counter()
    logits = np.concatenate(
        [predict_batch(test_texts[i : i + EVAL_BATCH_SIZE]) for i in range(0, len(test_texts), EVAL_BATCH_SIZE)]
    )
    sync_device()
    row["infer_throughput_sps"] = len(test_texts) / (time.perf_counter() - t0)
    row["infer_latency_ms"] = _time_latency(
        lambda t: predict_batch([t]), test_texts[: s.latency_samples]
    )

    mean_test_tokens = (
        textcnn.count_tokens(cnn.nlp, test_texts) / len(test_texts) if cnn else 0.0
    )
    row["infer_flops_per_sample"] = bert_fwd_flops + (
        cnn.flops_per_token * mean_test_tokens if cnn else 0.0
    )

    predictions = bert.to_frame(logits, ctx.test_df)
    row.update(_metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()))

    # -------------------------------------------------------------- totais
    row["train_time_s"] = row["cnn_train_s"] + row["cnn_oof_s"] + row["cnn_feat_s"] + row["bert_train_s"]
    row["total_time_s"] = row["train_time_s"] + row["generation_time_s"]
    row["train_flops"] = row["cnn_train_flops"] + row["bert_train_flops"]
    row["total_flops"] = row["train_flops"] + row["generation_flops"]
    if math.isnan(row["generation_time_s"]):
        logger.warning(
            "Tempo de geração não medido — total_time_s fica NaN para %s. "
            "Rode measure_generation_cost.py ou defina GENERATION_SEC_PER_SENTENCE.",
            arm,
        )

    del model
    _free_memory()
    return row, predictions
