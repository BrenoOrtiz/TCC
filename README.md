# Transfer learning TextCNN → TinyBERT vs TinyBERT + data augmentation

**Pergunta de pesquisa:** *transfer learning oferece custo-benefício
(performance / custo computacional) para a classificação de texto financeiro com
dados escassos?*

Continuação de [Data-Augmentation-Ollama](https://github.com/BrenoOrtiz/Data-Augmentation-Ollama):
mesmo dataset (Financial PhraseBank 75% Agree), mesmo split, mesmo test set fixo,
mesmos cenários de escassez e mesmos hiperparâmetros do TinyBERT. Só os dados do
LLM **llama3.1:latest** foram baixados (melhor resultado augmented no experimento
de origem: F1 macro 0.8912 em 25%, e CSVs sem textos multilinha).

---

## Desenho experimental

### Braços (3 por cenário de ratio)

| Braço | Pipeline | Dados de treino | Papel |
|---|---|---|---|
| **`tinybert_augmented`** | llama3.1 gera r·N frases → TinyBERT | (1−r)·N real + r·N sintético **gerado no run** | **braço DA** |
| `tinybert_restricted` | TinyBERT | (1−r)·N real | referência de escassez |
| **`hybrid_restricted`** | TextCNN → top-12 features → TinyBERT | (1−r)·N real | **braço TL** |

r ∈ {10%, 25%, 50%} disponíveis (o 75% não foi gerado no repositório de origem);
padrão atual `RATIOS = [10]`. Seeds {42, 43, 44} → **9 runs por cenário**, todos
avaliados no mesmo `data/raw/test.csv` (690 frases). Braços extras continuam
suportados via `--arms` (`tinybert_full`, `textcnn_restricted`, `hybrid_augmented`).

### Modelo híbrido (transfer learning por feature extraction)

```
texto ──► TextCNN spaCy (TextCatCNN.v2 / HashEmbedCNN.v2, treinada nos dados escassos)
            └─► 195 features: [mean-pool CNN (96) | max-pool CNN (96) | probs CNN (3)]
                  └─► seleção: top-12 por informação mútua com o rótulo (só no treino)
                        └─► LayerNorm → Linear(12→312) → GELU ─┐
texto ──► TinyBERT (4L-312D) ─► pooler[CLS] (312) ──────────────────┴─► concat (624) → dropout → Linear → 3 classes
```

- A CNN aprende padrões locais (n-gramas) de sentimento financeiro e **transfere**
  esse conhecimento ao TinyBERT como features; o TinyBERT é fine-tuned junto com
  a cabeça de fusão.
- **Top-12 features (`CNN_TOP_K_FEATURES = 12`):** as 195 features são ranqueadas
  por `mutual_info_classif` sobre as features out-of-fold do **treino**; os mesmos
  12 índices são aplicados ao teste. O teste nunca participa da seleção. As
  features escolhidas por seed ficam em `runs.csv` (`selected_features`) e no `report.md`.
- **Cross-fitting (out-of-fold, k=5):** as features de treino do híbrido vêm de
  CNNs que *não viram* aquela frase. Sem isso, a CNN superajustada entregaria
  probabilidades quase perfeitas no treino e o TinyBERT (e a seleção de features)
  confiaria nelas cegamente (vazamento clássico de stacking). O custo das k CNNs
  extras entra na conta do braço TL. `CNN_OOF_FOLDS = 0` desliga (ablação).
- `TinyBertClassifier(feature_dim=0)` é idêntico ao `BertForSequenceClassification`
  usado no experimento de origem; **a única diferença entre TinyBERT puro e
  híbrido é a entrada de features** — mesma inicialização, otimizador, épocas,
  LR, batch e seed.

### Custo computacional medido por run

| Métrica | Como |
|---|---|
| `train_time_s` | wall-clock de CNN (final + folds OOF) + fine-tuning TinyBERT, com `synchronize()` do MPS/CUDA |
| `generation_time_s` | wall-clock real da geração das r·N frases no Ollama (`LIVE_GENERATION = True`), inclui retries |
| `total_time_s` | treino + geração |
| `train_flops` | TinyBERT: FLOPs de forward contados pelo `torch.utils.flop_counter` × 3 (fwd+bwd) × amostras × épocas; CNN: 2·pesos densos por token × tokens × épocas × 3 |
| `generation_flops` | 2 · 8,03 B parâmetros · (tokens de prompt + tokens gerados), estimado pelos textos sintéticos |
| `total_flops` | treino + geração — **independente de hardware, sempre disponível** |
| `infer_latency_ms` | mediana, batch=1, ponta-a-ponta (texto cru → classe; inclui a CNN no híbrido) |
| `infer_throughput_sps`, `infer_flops_per_sample` | inferência no test set inteiro |
| `params_total`, `peak_accel_mb`, `peak_rss_mb`, `train_cpu_s` | tamanho do modelo e memória/CPU (indicativos) |

Um treino descartável aquece o backend antes do 1º run medido.

### Como a pergunta é respondida

Contraste principal **TL (`hybrid_restricted`) vs DA (`tinybert_augmented`)**, por nível de escassez:

1. **ΔF1 macro** médio entre seeds, com **IC 95% por bootstrap pareado** nas frases de teste
   (2000 reamostragens), + Wilcoxon pareado sobre todos os (ratio, seed).
2. **ΔCusto** em 4 visões: `total_flops`, `total_time_s` (DA paga a geração),
   `train_flops`, `train_time_s` (dados sintéticos tratados como já existentes).
3. **Veredito** (margem de não-inferioridade = 1 p.p. de F1):

| Situação | Veredito |
|---|---|
| IC > 0 e mais barato | **domina** |
| IC inferior > −1 p.p. e mais barato | **custo-efetivo** (não-inferior e mais barato) |
| IC > 0 e mais caro | trade-off — ver `Δcusto por p.p.` (análogo ao ICER) |
| mais barato, IC < 0 | trade-off (inferior) |
| mais barato, IC cruza −1 p.p. | inconclusivo (não-inferioridade não demonstrada) |
| IC < 0 e mais caro | dominado |

Contrastes secundários isolam os efeitos: TL vs TinyBERT escasso (ganho do TL) e
DA vs TinyBERT escasso (ganho do DA).
`efficiency.csv` traz p.p. de F1 ganhos por PFLOP / minuto extra em relação ao TinyBERT escasso.

---

## Estrutura

```
├── config.py                    # tudo configurável (braços, seeds, hiperparâmetros, custos)
├── run_experiment.py            # executa todos os runs + análise (retoma se interrompido)
├── analyze.py                   # refaz só a análise a partir de results/runs.csv
├── measure_generation_cost.py   # mede s/frase do llama3.1 (só p/ LIVE_GENERATION = False)
├── data/
│   ├── raw/                     # Sentences_75Agree.txt, train_full.csv, test.csv
│   ├── generated/llama3.1-latest/  # train_{restricted,augmented}_*, synthetic_only_*
│   └── reference/finetuning_results_original.csv  # resultados do experimento de origem
└── src/
    ├── data.py                  # leitura dos CSVs
    ├── textcnn.py               # TextCNN spaCy: treino, features, cross-fitting
    ├── bert.py                  # TinyBERT puro/híbrido, treino (HF Trainer), inferência
    ├── cost.py                  # monitor de tempo/memória, FLOPs
    ├── generation.py            # custo da geração sintética (FLOPs/tempo)
    ├── runner.py                # um run: treino, avaliação, custos
    └── analysis.py              # estatística, veredito, gráficos, report.md
```

## Como rodar

```bash
uv sync

# 1. teste rápido do pipeline (subamostra, 1 época) → results_smoke/
uv run python run_experiment.py --smoke

# 2. cenário(s) de RATIOS em config.py (padrão: 10%) → results/
uv run python run_experiment.py
uv run python run_experiment.py --ratios 10 25 50   # todos os cenários

```

**Pré-requisito do braço DA:** Ollama ativo com o modelo baixado
(`ollama serve` e `ollama pull llama3.1:latest`). Com `LIVE_GENERATION = True`
(padrão), cada seed do braço DA gera suas próprias frases sintéticas
(mesmos prompts, `temperature=0.8`, `num_predict=120` e cota por classe do
repositório de origem; `seed` do Ollama por chamada para reprodutibilidade),
cronometra a geração e treina o TinyBERT no dataset restrito + sintético novo.
As frases e o tempo medido ficam em `data/generated/llama3.1-latest-live/`
(`synthetic_<r>pct_seed<s>.csv/.json`) e são reaproveitados se o run for retomado.
`LIVE_GENERATION = False` volta a usar os CSVs baixados (tempo de geração
estimado por `measure_generation_cost.py`, se existir).

Subconjuntos: `--ratios 50 --seeds 42 --arms hybrid_restricted tinybert_augmented`.
Resultados de desenhos anteriores estão em `results_archive/` (10% parcial com 195 features; DA de 10% com os CSVs baixados).
O dispositivo é escolhido automaticamente (CUDA > MPS > CPU); force com `DEVICE` em `config.py`.
**Compare custos de tempo apenas entre runs feitos na mesma máquina.**

## Saídas (`results/`)

| Arquivo | Conteúdo |
|---|---|
| `runs.csv` | uma linha por run com todas as métricas de qualidade e custo |
| `predictions/<braço>__<ratio>__<seed>.csv` | predições e probabilidades no test set |
| `summary.csv` | média ± dp por (braço, ratio) |
| `contrasts.csv` | ΔF1 + IC, Δcusto, razão de custo, custo por p.p., veredito |
| `efficiency.csv` | ganho de F1 por custo extra vs TinyBERT escasso |
| `report.md` | **relatório que responde à pergunta** |
| `fig_f1_vs_cost_*.png` | F1 × custo (log) com fronteira de Pareto, um painel por ratio |
| `fig_f1_by_ratio.png`, `fig_cost_breakdown.png`, `fig_delta_f1_forest.png` | F1 por escassez, composição do custo, ΔF1 com IC |

## Limitações a declarar no TCC

- O braço DA usa sintéticos **gerados ao vivo** (não os CSVs do repositório de
  origem), então seu F1 não é idêntico ao do experimento original. A carga do
  modelo no Ollama fica fora do cronômetro (análogo ao warm-up do TinyBERT).
- A CNN do spaCy roda em CPU e o TinyBERT em GPU/MPS: o tempo reflete o pipeline
  real, mas não é um custo "normalizado" — por isso os FLOPs são reportados junto.
- FLOPs são estimativas analíticas (tokens por caractere ≈ 4 nas frases
  sintéticas; FLOPs da CNN ≈ 2 × pesos densos por token).
- 3 seeds × 690 frases de teste: diferenças < ~1 p.p. tendem a ficar dentro do IC.
- MPS não é totalmente determinístico; seeds reduzem, mas não eliminam, a variação.
