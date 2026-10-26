"""Gera data/generated/<llm>/train_restricted_<r>pct.csv com o mesmo método do
repositório de origem (src/data/restrictor.py de Data-Augmentation-Ollama):
amostragem estratificada por classe, keep_frac = 1 − r, random_state = 42.

Só o 75% precisou ser recriado (os CSVs de 10/25/50% foram baixados); o script
confere que reproduz os arquivos existentes byte a byte antes de escrever.

Uso:
    uv run python make_restricted_split.py          # 75%
    uv run python make_restricted_split.py 75 90
"""

import sys

import pandas as pd

from config import AUGMENTATION_LLM_SLUG, GENERATED_DIR, RAW_DIR

ORIGIN_SEED = 42


def restrict(train_full: pd.DataFrame, ratio: int) -> pd.DataFrame:
    keep_frac = 1.0 - ratio / 100
    return (
        train_full.groupby("label", group_keys=False)[train_full.columns.tolist()]
        .apply(lambda g: g.sample(frac=keep_frac, random_state=ORIGIN_SEED))
        .reset_index(drop=True)[["text", "sentiment"]]
    )


def main(ratios: list[int]) -> None:
    train_full = pd.read_csv(RAW_DIR / "train_full.csv")
    out_dir = GENERATED_DIR / AUGMENTATION_LLM_SLUG
    for path in sorted(out_dir.glob("train_restricted_*pct.csv")):
        ratio = int(path.stem.split("_")[-1].removesuffix("pct"))
        if ratio not in ratios and not restrict(train_full, ratio).equals(pd.read_csv(path)):
            raise SystemExit(f"Método não reproduz {path.name}; abortando.")
    for ratio in ratios:
        df = restrict(train_full, ratio)
        df.to_csv(out_dir / f"train_restricted_{ratio}pct.csv", index=False)
        print(f"{ratio}%: {len(df)} frases reais {df['sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main([int(r) for r in sys.argv[1:]] or [75])
