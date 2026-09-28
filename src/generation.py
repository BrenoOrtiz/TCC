"""Geração sintética e seu custo (braços *_augmented).

Os prompts, a temperatura, o limite de tokens e a cota por classe são os mesmos
de `src/augmentation/` do repositório Data-Augmentation-Ollama.

- `generate_live`: gera as frases via Ollama e cronometra (custo real).
- `generation_cost`: estima o custo dos CSVs baixados (sem tempo registrado).
"""

import json
import logging
import math
import time
from dataclasses import dataclass

import pandas as pd

from config import (
    AUGMENTATION_LLM,
    AUGMENTATION_LLM_PARAMS,
    CHARS_PER_TOKEN,
    GEN_MAX_RETRIES,
    GEN_MAX_TOKENS,
    GEN_TEMPERATURE,
    GENERATION_SEC_PER_SENTENCE,
    LABEL_NAMES,
    LIVE_GENERATED_DIR,
    OLLAMA_HOST,
    RESULTS_DIR,
)
from src.cost import llm_generation_flops

logger = logging.getLogger(__name__)

GENERATION_COST_FILE = RESULTS_DIR / "generation_cost.json"
CHAT_TEMPLATE_OVERHEAD_TOKENS = 20

SYSTEM_PROMPT = (
    "You are a financial journalist. Your task is to generate realistic, concise "
    "financial news sentences in English, similar to the Financial PhraseBank dataset. "
    "Write exactly one sentence per response. Do not include labels, explanations, "
    "quotes, or any additional text."
)

USER_PROMPTS: dict[str, str] = {
    "positive": (
        "Generate one short financial news sentence with POSITIVE sentiment. "
        "Examples of positive topics: earnings beat, revenue growth, profit increase, "
        "successful acquisition, raised guidance, dividend increase, stock rally."
    ),
    "neutral": (
        "Generate one short financial news sentence with NEUTRAL sentiment. "
        "Examples of neutral topics: scheduled filings, management appointments, "
        "operational announcements, product launches with no clear financial impact, "
        "routine business updates."
    ),
    "negative": (
        "Generate one short financial news sentence with NEGATIVE sentiment. "
        "Examples of negative topics: profit warning, revenue decline, layoffs, "
        "failed deal, debt downgrade, stock drop, missed earnings, legal setback."
    ),
}


def sec_per_sentence() -> float | None:
    """Medição real (measure_generation_cost.py) > valor do config > None."""
    if GENERATION_COST_FILE.exists():
        return float(json.loads(GENERATION_COST_FILE.read_text())["sec_per_sentence"])
    return GENERATION_SEC_PER_SENTENCE


def generation_cost(synthetic_df: pd.DataFrame) -> tuple[float, float]:
    """(FLOPs estimados, segundos) para gerar `synthetic_df`. Segundos = NaN se não medido."""
    flops = 0.0
    for sentiment, group in synthetic_df.groupby("sentiment"):
        prompt_chars = len(SYSTEM_PROMPT) + len(USER_PROMPTS[sentiment])
        prompt_tokens = prompt_chars / CHARS_PER_TOKEN + CHAT_TEMPLATE_OVERHEAD_TOKENS
        output_tokens = group["text"].str.len() / CHARS_PER_TOKEN
        flops += float(
            llm_generation_flops(AUGMENTATION_LLM_PARAMS, prompt_tokens, output_tokens).sum()
        )
    sps = sec_per_sentence()
    seconds = sps * len(synthetic_df) if sps is not None else float("nan")
    return flops, seconds


# ---------------------------------------------------------------------------
# Geração ao vivo
# ---------------------------------------------------------------------------

@dataclass
class LiveGeneration:
    synthetic: pd.DataFrame
    seconds: float          # wall-clock da geração (inclui retries e respostas vazias)
    prompt_tokens: int      # contados pelo Ollama
    output_tokens: int
    calls: int

    @property
    def flops(self) -> float:
        return llm_generation_flops(AUGMENTATION_LLM_PARAMS, self.prompt_tokens, self.output_tokens)


def synthetic_quota(train_full: pd.DataFrame, ratio: int) -> dict[str, int]:
    """Cota por classe = ratio·N distribuído pela proporção original (igual ao repo de origem)."""
    total = math.ceil(len(train_full) * ratio / 100)
    proportions = train_full["sentiment"].value_counts(normalize=True)
    return {sentiment: max(1, math.ceil(total * prop)) for sentiment, prop in proportions.items()}


def _clean(text: str) -> str:
    return text.strip().strip("\"'").strip()


def _paths(ratio: int, seed: int):
    stem = LIVE_GENERATED_DIR / f"synthetic_{ratio}pct_seed{seed}"
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def generate_live(train_full: pd.DataFrame, ratio: int, seed: int) -> LiveGeneration:
    """Gera a cota sintética com o LLM e cronometra.

    Cada seed é uma replicação independente do pipeline DA (gera seus próprios
    dados). O resultado é salvo em data/generated/<llm>-live/; se já existir,
    é reaproveitado com o tempo medido originalmente (permite retomar runs).
    """
    csv_path, meta_path = _paths(ratio, seed)
    if csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        logger.info("  Reaproveitando geração salva em %s (%.1fs)", csv_path.name, meta["seconds"])
        return LiveGeneration(pd.read_csv(csv_path), meta["seconds"], meta["prompt_tokens"],
                              meta["output_tokens"], meta["calls"])

    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    # carrega o modelo na memória antes do cronômetro (equivalente ao warm-up do TinyBERT)
    client.chat(model=AUGMENTATION_LLM, messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1})

    quota = synthetic_quota(train_full, ratio)
    logger.info("  Gerando %d frases com %s %s", sum(quota.values()), AUGMENTATION_LLM, quota)
    records, prompt_tokens, output_tokens, calls = [], 0, 0, 0
    t0 = time.perf_counter()
    for sentiment, count in quota.items():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPTS[sentiment]},
        ]
        generated, failures = 0, 0
        while generated < count and failures < count * 3:
            calls += 1
            text = None
            for attempt in range(GEN_MAX_RETRIES):
                try:
                    resp = client.chat(
                        model=AUGMENTATION_LLM,
                        messages=messages,
                        # seed por chamada: saídas variadas e reprodutíveis
                        options={"temperature": GEN_TEMPERATURE, "num_predict": GEN_MAX_TOKENS,
                                 "seed": seed * 1_000_000 + calls},
                    )
                    prompt_tokens += resp.prompt_eval_count or 0
                    output_tokens += resp.eval_count or 0
                    text = _clean(resp.message.content)
                    break
                except Exception as exc:  # noqa: BLE001 — mesmo tratamento do repo de origem
                    logger.warning("  Falha %d/%d (%s): %s", attempt + 1, GEN_MAX_RETRIES, sentiment, exc)
                    time.sleep(2**attempt)
            if text:
                records.append({"text": text, "label": LABEL_NAMES[sentiment], "sentiment": sentiment,
                                "source": "synthetic", "model": AUGMENTATION_LLM})
                generated += 1
            else:
                failures += 1
    seconds = time.perf_counter() - t0

    synthetic = pd.DataFrame(records)
    LIVE_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({
        "model": AUGMENTATION_LLM, "ratio": ratio, "seed": seed, "n": len(synthetic),
        "seconds": seconds, "sec_per_sentence": seconds / max(len(synthetic), 1),
        "prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "calls": calls,
    }, indent=2))
    logger.info("  %d frases em %.1fs (%.2fs/frase)", len(synthetic), seconds, seconds / max(len(synthetic), 1))
    return LiveGeneration(synthetic, seconds, prompt_tokens, output_tokens, calls)
