"""TextCNN do spaCy: treino, predição e extração de features para o TinyBERT.

Arquitetura: `spacy.TextCatCNN.v2` sobre `spacy.HashEmbedCNN.v2`
    tokens -> hash embeddings -> 4 blocos CNN (maxout, janela 3) -> mean pool -> softmax

Features exportadas por frase (dimensão 2·width + n_classes):
    [mean-pool dos vetores CNN | max-pool dos vetores CNN | probabilidades da CNN]
e, para o TinyBERT, apenas as K mais importantes (`select_top_features`).
"""

import logging
import random
import time
from dataclasses import dataclass

import numpy as np
import spacy
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, minibatch

from config import (
    CNN_BATCH_COMPOUND,
    CNN_BATCH_START,
    CNN_BATCH_STOP,
    CNN_DEPTH,
    CNN_DROPOUT,
    CNN_EMBED_SIZE,
    CNN_EPOCHS,
    CNN_MAXOUT_PIECES,
    CNN_WIDTH,
    CNN_WINDOW_SIZE,
    LABEL_NAMES,
)

logger = logging.getLogger(__name__)

LABELS = sorted(LABEL_NAMES, key=LABEL_NAMES.get)  # ordem = id do rótulo
FEATURE_DIM = 2 * CNN_WIDTH + len(LABELS)
FEATURE_NAMES = (
    [f"mean_{i}" for i in range(CNN_WIDTH)]
    + [f"max_{i}" for i in range(CNN_WIDTH)]
    + [f"prob_{label}" for label in LABELS]
)


@dataclass
class CNNTraining:
    nlp: Language
    train_s: float
    n_params: int
    flops_per_token: float


def _build(seed: int) -> Language:
    spacy.util.fix_random_seed(seed)
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe(
        "textcat",
        config={
            "model": {
                "@architectures": "spacy.TextCatCNN.v2",
                "exclusive_classes": True,
                "nO": None,
                "tok2vec": {
                    "@architectures": "spacy.HashEmbedCNN.v2",
                    "pretrained_vectors": None,
                    "width": CNN_WIDTH,
                    "depth": CNN_DEPTH,
                    "embed_size": CNN_EMBED_SIZE,
                    "window_size": CNN_WINDOW_SIZE,
                    "maxout_pieces": CNN_MAXOUT_PIECES,
                    "subword_features": True,
                },
            }
        },
    )
    for label in LABELS:
        textcat.add_label(label)
    return nlp


def _examples(nlp: Language, texts: list[str], labels: list[int]) -> list[Example]:
    return [
        Example.from_dict(
            nlp.make_doc(text),
            {"cats": {name: float(i == label) for i, name in enumerate(LABELS)}},
        )
        for text, label in zip(texts, labels)
    ]


def _n_params(nlp: Language) -> int:
    model = nlp.get_pipe("textcat").model
    return int(
        sum(
            node.get_param(name).size
            for node in model.walk()
            for name in node.param_names
            if node.has_param(name)
        )
    )


def _flops_per_token(nlp: Language) -> float:
    """Estimativa: 2 FLOPs por peso das camadas densas (maxout/softmax).

    Hash embeddings são lookups (sem multiplicação) e ficam de fora; o softmax
    roda uma vez por documento, mas é desprezível e é contado por token.
    """
    model = nlp.get_pipe("textcat").model
    dense = sum(
        node.get_param("W").size
        for node in model.walk()
        if node.name in {"maxout", "softmax"} and node.has_param("W")
    )
    return 2.0 * float(dense)


def train_textcnn(
    texts: list[str], labels: list[int], seed: int, epochs: int = CNN_EPOCHS
) -> CNNTraining:
    nlp = _build(seed)
    rng = random.Random(seed)
    t0 = time.perf_counter()
    examples = _examples(nlp, texts, labels)
    optimizer = nlp.initialize(lambda: examples)
    for epoch in range(epochs):
        rng.shuffle(examples)
        losses: dict[str, float] = {}
        sizes = compounding(CNN_BATCH_START, CNN_BATCH_STOP, CNN_BATCH_COMPOUND)
        for batch in minibatch(examples, size=sizes):
            nlp.update(batch, drop=CNN_DROPOUT, sgd=optimizer, losses=losses)
        logger.debug("  CNN epoch %d loss=%.4f", epoch + 1, losses.get("textcat", 0.0))
    train_s = time.perf_counter() - t0
    return CNNTraining(nlp, train_s, _n_params(nlp), _flops_per_token(nlp))


def predict_proba(nlp: Language, texts: list[str]) -> np.ndarray:
    model = nlp.get_pipe("textcat").model
    docs = [nlp.make_doc(t) for t in texts]
    return model.ops.to_numpy(model.predict(docs)).astype(np.float32)


def extract_features(nlp: Language, texts: list[str]) -> np.ndarray:
    """Vetor por frase: mean-pool, max-pool dos vetores CNN e probabilidades."""
    model = nlp.get_pipe("textcat").model
    tok2vec = model.get_ref("tok2vec")
    docs = [nlp.make_doc(t) for t in texts]
    token_vectors = tok2vec.predict(docs)
    probs = model.ops.to_numpy(model.predict(docs))

    feats = np.zeros((len(docs), FEATURE_DIM), dtype=np.float32)
    for i, vecs in enumerate(token_vectors):
        vecs = model.ops.to_numpy(vecs)
        if len(vecs):
            feats[i, :CNN_WIDTH] = vecs.mean(axis=0)
            feats[i, CNN_WIDTH : 2 * CNN_WIDTH] = vecs.max(axis=0)
        feats[i, 2 * CNN_WIDTH :] = probs[i]
    return feats


def out_of_fold_features(
    texts: list[str], labels: list[int], seed: int, folds: int, epochs: int = CNN_EPOCHS
) -> tuple[np.ndarray, float]:
    """Cross-fitting: cada frase de treino recebe features de uma CNN que não a viu."""
    feats = np.zeros((len(texts), FEATURE_DIM), dtype=np.float32)
    texts_arr, labels_arr = np.array(texts, dtype=object), np.array(labels)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    for fold, (fit_idx, held_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        fold_cnn = train_textcnn(
            texts_arr[fit_idx].tolist(), labels_arr[fit_idx].tolist(), seed + fold, epochs
        )
        feats[held_idx] = extract_features(fold_cnn.nlp, texts_arr[held_idx].tolist())
    return feats, time.perf_counter() - t0


def select_top_features(
    features: np.ndarray, labels: list[int], k: int, seed: int
) -> np.ndarray:
    """Índices das k features com maior informação mútua com o rótulo (ordem decrescente).

    Deve receber features de TREINO (out-of-fold), nunca do teste.
    """
    scores = mutual_info_classif(features, np.asarray(labels), random_state=seed)
    return np.argsort(scores)[::-1][:k]


def count_tokens(nlp: Language, texts: list[str]) -> int:
    return sum(len(nlp.make_doc(t)) for t in texts)
