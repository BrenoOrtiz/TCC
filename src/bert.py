"""TinyBERT puro e híbrido (TextCNN -> TinyBERT), com a mesma cabeça e treino.

A única diferença entre os dois é a entrada `cnn_features`:

    TinyBERT puro : pooler[CLS] -> dropout -> Linear(h, 3)
    Híbrido       : [pooler[CLS] ; proj(features CNN)] -> dropout -> Linear(2h, 3)

`TinyBertClassifier(feature_dim=0)` equivale a `BertForSequenceClassification`.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from config import (
    DEVICE,
    HEAD_DROPOUT,
    LABEL_NAMES,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    TINYBERT_MODEL,
    TRAIN_BATCH_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


class TinyBertClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int = 0, n_labels: int = len(LABEL_NAMES)):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.init_range = encoder.config.initializer_range
        self.feature_dim = feature_dim
        self.feature_proj = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
            )
            if feature_dim
            else None
        )
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.classifier = nn.Linear(hidden * (2 if feature_dim else 1), n_labels)
        self.reset_head()

    def reset_head(self) -> None:
        """Mesma inicialização do BERT (normal, std=initializer_range) em todas as cabeças."""
        heads = [self.classifier] + ([self.feature_proj[1]] if self.feature_proj else [])
        for layer in heads:
            nn.init.normal_(layer.weight, std=self.init_range)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cnn_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        if self.feature_proj is not None:
            pooled = torch.cat([pooled, self.feature_proj(cnn_features)], dim=-1)
        logits = self.classifier(self.dropout(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class EncodedDataset(Dataset):
    """Tokeniza tudo de uma vez (padding fixo em MAX_SEQ_LENGTH, como no experimento de origem)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: PreTrainedTokenizerBase,
        features: np.ndarray | None = None,
    ):
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.features = torch.from_numpy(features) if features is not None else None

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.features is not None:
            item["cnn_features"] = self.features[idx]
        return item


class PretrainedCache:
    """Carrega tokenizer + encoder uma única vez; cada run recebe uma cópia."""

    def __init__(self) -> None:
        logger.info("Carregando %s …", TINYBERT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(TINYBERT_MODEL)
        self._encoder = AutoModel.from_pretrained(TINYBERT_MODEL)

    def new_model(self, seed: int, feature_dim: int) -> TinyBertClassifier:
        set_seed(seed)
        return TinyBertClassifier(copy.deepcopy(self._encoder), feature_dim=feature_dim)


def resolve_device() -> torch.device:
    if DEVICE != "auto":
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model: TinyBertClassifier,
    dataset: EncodedDataset,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> TinyBertClassifier:
    device = resolve_device()
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_RATIO,
        eval_strategy="no",   # avaliação só no final, fora do tempo de treino
        save_strategy="no",
        logging_strategy="no",
        dataloader_pin_memory=device.type == "cuda",
        seed=seed,
        report_to="none",
        fp16=device.type == "cuda",
        use_cpu=device.type == "cpu",
        disable_tqdm=True,
    )
    Trainer(model=model, args=args, train_dataset=dataset).train()
    return model


@torch.no_grad()
def predict_logits(
    model: TinyBertClassifier,
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    features: np.ndarray | None = None,
) -> np.ndarray:
    """Inferência a partir do texto cru (tokenização incluída no custo)."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        ds = EncodedDataset(
            chunk,
            None,
            tokenizer,
            None if features is None else features[start : start + batch_size],
        )
        inputs = {
            "input_ids": ds.input_ids.to(device),
            "attention_mask": ds.attention_mask.to(device),
        }
        if ds.features is not None:
            inputs["cnn_features"] = ds.features.to(device)
        out.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(out)


def flops_inputs(tokenizer: PreTrainedTokenizerBase, feature_dim: int) -> dict[str, torch.Tensor]:
    """Um exemplo sintético de tamanho MAX_SEQ_LENGTH para contar FLOPs por amostra."""
    ds = EncodedDataset(["x"], None, tokenizer)
    inputs = {"input_ids": ds.input_ids, "attention_mask": ds.attention_mask}
    if feature_dim:
        inputs["cnn_features"] = torch.zeros(1, feature_dim)
    return inputs


def to_frame(logits: np.ndarray, test_df: pd.DataFrame) -> pd.DataFrame:
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    frame = pd.DataFrame({"y_true": test_df["label"].to_numpy(), "y_pred": probs.argmax(1)})
    for name, idx in LABEL_NAMES.items():
        frame[f"p_{name}"] = probs[:, idx]
    return frame
