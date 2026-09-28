import pandas as pd

from config import AUGMENTATION_LLM_SLUG, GENERATED_DIR, LABEL_NAMES, RAW_DIR


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas `text` (str) e `label` (int); CSVs restritos só têm `sentiment`."""
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df["sentiment"].map(LABEL_NAMES)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_test() -> pd.DataFrame:
    return _normalize(pd.read_csv(RAW_DIR / "test.csv"))


def load_train(variant: str, ratio: int | None) -> pd.DataFrame:
    """variant: full | restricted | augmented."""
    if variant == "full":
        return _normalize(pd.read_csv(RAW_DIR / "train_full.csv"))
    path = GENERATED_DIR / AUGMENTATION_LLM_SLUG / f"train_{variant}_{ratio}pct.csv"
    return _normalize(pd.read_csv(path))


def load_synthetic(ratio: int) -> pd.DataFrame:
    path = GENERATED_DIR / AUGMENTATION_LLM_SLUG / f"synthetic_only_{ratio}pct.csv"
    return _normalize(pd.read_csv(path))


def arm_data_variant(arm: str) -> str:
    """tinybert_full -> full, hybrid_augmented -> augmented, ..."""
    return arm.rsplit("_", 1)[1]


def arm_model(arm: str) -> str:
    """tinybert | textcnn | hybrid."""
    return arm.split("_", 1)[0]
