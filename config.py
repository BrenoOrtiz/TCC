"""Configuração centralizada do experimento.

Pergunta de pesquisa:
    Transfer learning (TextCNN spaCy -> TinyBERT) oferece melhor custo-benefício
    (performance / custo computacional) do que TinyBERT + data augmentation (LLM)
    para classificação de sentimento financeiro com dados escassos?
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GENERATED_DIR = DATA_DIR / "generated"
REFERENCE_DIR = DATA_DIR / "reference"
RESULTS_DIR = ROOT_DIR / "results"

# ---------------------------------------------------------------------------
# Dados (herdados do experimento Data-Augmentation-Ollama)
# Somente o LLM llama3.1:latest foi baixado: melhor resultado augmented
# (F1 macro 0.8912 em 25%) e CSVs sem textos multilinha.
# ---------------------------------------------------------------------------
AUGMENTATION_LLM = "llama3.1:latest"
AUGMENTATION_LLM_SLUG = "llama3.1-latest"
AUGMENTATION_LLM_PARAMS = 8.03e9  # usado na estimativa de FLOPs da geração

# 75%: train_restricted_75pct.csv recriado por make_restricted_split.py (mesmo método da origem)
RATIOS = [10, 25, 50, 75]

LABEL_NAMES = {"negative": 0, "neutral": 1, "positive": 2}
SEEDS = [42, 43, 44]

# ---------------------------------------------------------------------------
# Braços experimentais (3 por cenário de ratio)
#   tinybert_augmented   -> BRAÇO DA: dataset augmentado (real + sintético) + TinyBERT
#   tinybert_restricted  -> referência de escassez: dataset restrito + TinyBERT
#   hybrid_restricted    -> BRAÇO TL: dataset restrito + TextCNN (features) + TinyBERT
# Também suportados via --arms: tinybert_full (teto, ratio 0), textcnn_restricted,
# hybrid_augmented.
# ---------------------------------------------------------------------------
REFERENCE_ARMS: list[str] = []
RATIO_ARMS = [
    "tinybert_augmented",
    "tinybert_restricted",
    "hybrid_restricted",
]

# Contraste principal da pergunta de pesquisa: (tratamento, comparador)
PRIMARY_CONTRAST = ("hybrid_restricted", "tinybert_augmented")
SECONDARY_CONTRASTS = [
    ("hybrid_restricted", "tinybert_restricted"),   # ganho do TL sem DA
    ("tinybert_augmented", "tinybert_restricted"),  # ganho do DA
]

# ---------------------------------------------------------------------------
# TinyBERT (mesmos hiperparâmetros do experimento de origem, por paridade)
# ---------------------------------------------------------------------------
TINYBERT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
HEAD_DROPOUT = 0.1
DEVICE = "auto"  # auto | cpu | mps | cuda

# ---------------------------------------------------------------------------
# TextCNN spaCy (spacy.TextCatCNN.v2 + HashEmbedCNN.v2)
# ---------------------------------------------------------------------------
CNN_WIDTH = 96
CNN_DEPTH = 4
CNN_EMBED_SIZE = 2000
CNN_WINDOW_SIZE = 1
CNN_MAXOUT_PIECES = 3
CNN_EPOCHS = 10
CNN_DROPOUT = 0.2
CNN_BATCH_START = 4.0
CNN_BATCH_STOP = 32.0
CNN_BATCH_COMPOUND = 1.001

# Features de treino do híbrido são geradas out-of-fold (cross-fitting) para
# que o TinyBERT não aprenda a confiar em probabilidades superajustadas da CNN.
# 0 desativa (features in-sample; mais barato, porém enviesado).
CNN_OOF_FOLDS = 5

# Transfer learning por feature extraction: das 195 features da CNN
# (mean-pool 96 | max-pool 96 | probabilidades 3) só as K mais importantes vão
# para o TinyBERT. Importância = informação mútua com o rótulo, calculada nas
# features out-of-fold do treino (o teste nunca participa da seleção).
CNN_TOP_K_FEATURES = 12

# ---------------------------------------------------------------------------
# Medição de custo
# ---------------------------------------------------------------------------
LATENCY_SAMPLES = 200          # nº de frases do teste para latência batch=1
MONITOR_INTERVAL_S = 0.2       # amostragem de memória

# Custo da geração sintética (wall-clock). Se results/generation_cost.json
# existir (measure_generation_cost.py) ele tem precedência; senão usa este
# valor. None = não medido (a análise usa apenas FLOPs para o custo da geração).
GENERATION_SEC_PER_SENTENCE: float | None = None

# Geração ao vivo no braço DA: em cada seed o llama3.1 gera as frases sintéticas
# via Ollama (cronometrado) e o dataset augmentado = restrito + sintético novo.
# O custo do DA passa a ser tempo TOTAL real (geração + treino). False = usa os
# CSVs baixados do repositório de origem (tempo de geração estimado/ausente).
LIVE_GENERATION = True
OLLAMA_HOST = "http://localhost:11434"
GEN_TEMPERATURE = 0.8          # mesmos parâmetros do repositório de origem
GEN_MAX_TOKENS = 120
GEN_MAX_RETRIES = 3
LIVE_GENERATED_DIR = GENERATED_DIR / f"{AUGMENTATION_LLM_SLUG}-live"
CHARS_PER_TOKEN = 4.0          # estimativa de tokens das frases sintéticas

# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 2000
NON_INFERIORITY_MARGIN = 0.01  # 1 p.p. de F1 macro
# total_* inclui a geração sintética no custo do DA; train_* ignora (dados já gerados)
COST_METRICS = ["total_time_s", "total_flops", "train_time_s", "train_flops"]
