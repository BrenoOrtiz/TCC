"""Mede o custo real (wall-clock) de gerar frases sintéticas com o LLM do DA.

O tempo de geração dos CSVs originais não foi registrado no repositório de
origem; este script reproduz a mesma chamada (mesmos prompts, temperatura e
limite de tokens) numa amostra balanceada e grava results/generation_cost.json,
usado pela análise para compor `total_time_s` dos braços *_augmented.

Rode na MESMA máquina do experimento, com o Ollama ativo:
    ollama serve
    ollama pull llama3.1:latest
    uv run python measure_generation_cost.py --n 60
"""

import argparse
import json
import platform
import statistics
import time

import ollama

from config import AUGMENTATION_LLM, RESULTS_DIR
from src.generation import GENERATION_COST_FILE, SYSTEM_PROMPT, USER_PROMPTS

TEMPERATURE = 0.8
MAX_TOKENS = 120
OLLAMA_HOST = "http://localhost:11434"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="nº de frases (divididas entre as classes)")
    args = parser.parse_args()

    client = ollama.Client(host=OLLAMA_HOST)
    # 1ª chamada carrega o modelo na memória; não entra na medição
    client.chat(model=AUGMENTATION_LLM, messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1})

    sentiments = list(USER_PROMPTS)
    durations, prompt_tokens, output_tokens = [], [], []
    for i in range(args.n):
        sentiment = sentiments[i % len(sentiments)]
        t0 = time.perf_counter()
        resp = client.chat(
            model=AUGMENTATION_LLM,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPTS[sentiment]},
            ],
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        )
        durations.append(time.perf_counter() - t0)
        prompt_tokens.append(resp.get("prompt_eval_count") or 0)
        output_tokens.append(resp.get("eval_count") or 0)
        print(f"[{i + 1}/{args.n}] {sentiment:8s} {durations[-1]:.2f}s  {resp['message']['content'][:70]!r}")

    result = {
        "model": AUGMENTATION_LLM,
        "n": args.n,
        "sec_per_sentence": statistics.mean(durations),
        "sec_per_sentence_median": statistics.median(durations),
        "mean_prompt_tokens": statistics.mean(prompt_tokens),
        "mean_output_tokens": statistics.mean(output_tokens),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATION_COST_FILE.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nGravado em {GENERATION_COST_FILE}. Rode `uv run python analyze.py` para atualizar a análise.")


if __name__ == "__main__":
    main()
