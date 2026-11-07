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

r ∈ {10%, 25%, 50%, 75%} (`RATIOS` em `config.py`). Os splits restritos de 10/25/50%
vieram do repositório de origem; o de 75% (689 frases reais) é recriado por
`make_restricted_split.py` com o mesmo método da origem (amostragem estratificada por
classe, `random_state=42`; o script confere que reproduz os outros três arquivos
exatamente). Seeds {42, 43, 44} → **9 runs por cenário** (36 no total), todos
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

## Conteúdo do material entregue

Mapa de **todos** os arquivos e pastas do projeto, agrupados por tipo. Caminhos
relativos à raiz do repositório.

### Visão geral

```
TCC/
├── README.md                    # este documento
├── config.py                    # configuração central do experimento
├── run_experiment.py            # executa os runs + análise
├── analyze.py                   # refaz só a análise a partir de results/runs.csv
├── make_restricted_split.py     # recria os splits restritos (usado para 75%)
├── measure_generation_cost.py   # mede s/frase do llama3.1 (modo sem geração ao vivo)
├── pyproject.toml, uv.lock      # dependências (ambiente reprodutível com uv)
├── .python-version              # versão do Python (3.13)
├── .gitignore                   # exclui .venv, caches, results_smoke/ e results/_work/
├── src/                         # código-fonte do pipeline
├── data/                        # dados brutos, splits, sintéticos e referência
├── results/                     # resultados finais (36 runs): métricas, predições, figuras, relatórios
├── results_archive/             # resultados de desenhos anteriores (não usados nas conclusões)
└── logs/                        # logs de execução dos experimentos
```

### 1. Código-fonte

| Arquivo | Função |
|---|---|
| `config.py` | Todos os parâmetros: caminhos, cenários (`RATIOS`), seeds, braços, hiperparâmetros do TinyBERT e da TextCNN, geração ao vivo, bootstrap, margem de não-inferioridade, tempo de anotação humana (`HUMAN_ANNOTATION_SEC_PER_SAMPLE = 4.6` s/título) |
| `run_experiment.py` | Ponto de entrada: executa os runs (braço × ratio × seed), grava `results/runs.csv` e `results/predictions/`, retoma runs interrompidos e chama a análise no final |
| `analyze.py` | Refaz só estatística, gráficos e relatório a partir de `results/runs.csv` (não treina nada) |
| `make_restricted_split.py` | Recria `data/generated/llama3.1-latest/train_restricted_<r>pct.csv` pelo método da origem (estratificado, `random_state=42`); confere que reproduz 10/25/50% e gera o de 75% |
| `measure_generation_cost.py` | Mede o tempo por frase do llama3.1 no Ollama (só usado com `LIVE_GENERATION = False`) |
| `src/__init__.py` | Marca `src/` como pacote Python (vazio) |
| `src/data.py` | Leitura dos CSVs de treino e teste |
| `src/textcnn.py` | TextCNN spaCy (`TextCatCNN.v2` + `HashEmbedCNN.v2`): treino, extração das 195 features, cross-fitting out-of-fold |
| `src/bert.py` | TinyBERT puro e híbrido (fusão com as features da CNN), treino via HF `Trainer`, inferência |
| `src/generation.py` | Geração sintética ao vivo com llama3.1 (Ollama) e estimativa de FLOPs/tempo da geração |
| `src/cost.py` | Medição de custo: tempo wall-clock, CPU, memória, contagem de FLOPs |
| `src/runner.py` | Executa um run: prepara dados, treina, avalia no teste, mede custos |
| `src/analysis.py` | Estatística (bootstrap pareado, Wilcoxon), veredito custo-efetividade, todos os gráficos e o `report.md` |

### 2. Dados (`data/`)

| Caminho | Conteúdo |
|---|---|
| `data/raw/Sentences_75Agree.txt` | Financial PhraseBank original, subconjunto com ≥ 75% de concordância entre anotadores (3453 frases, formato `frase@rótulo`) |
| `data/raw/train_full.csv` | Split de treino completo (2758 frases reais) — colunas `text, sentiment, label` |
| `data/raw/test.csv` | Test set fixo usado em todos os runs (690 frases) |
| `data/generated/llama3.1-latest/train_restricted_{10,25,50,75}pct.csv` | Treino escasso: 2482 / 2069 / 1378 / 689 frases reais (10–50% vindos do repositório de origem; 75% recriado por `make_restricted_split.py`) |
| `data/generated/llama3.1-latest/train_augmented_{10,25,50}pct.csv`, `synthetic_only_{10,25,50}pct.csv` | Datasets augmentados e frases sintéticas **baixados** do repositório de origem; usados só com `LIVE_GENERATION = False` e no arquivo `results_archive/10pct_da_csv_baixado/` |
| `data/generated/llama3.1-latest-live/synthetic_<r>pct_seed<s>.csv` | Frases sintéticas **geradas ao vivo** pelo llama3.1 em cada run do braço DA (4 ratios × 3 seeds = 12 arquivos) — são as usadas nos resultados finais |
| `data/generated/llama3.1-latest-live/synthetic_<r>pct_seed<s>.json` | Metadados de cada geração: nº de frases, tempo medido (s), s/frase, tokens de prompt e de saída, nº de chamadas |
| `data/reference/finetuning_results_original.csv` | Resultados do experimento de origem (Data-Augmentation-Ollama), inclusive o baseline TinyBERT com 100% dos dados (F1 macro 0,893) |

### 3. Modelos treinados

**Nenhum peso de modelo treinado é salvo em disco.** O treino usa
`save_strategy="no"` (`src/bert.py`) e a TextCNN existe só em memória durante o
run; cada um dos 36 runs treina seus modelos do zero e grava apenas métricas e
predições. Para reconstruir qualquer modelo, rode novamente o run
correspondente (mesmo braço, ratio e seed — ver "Como rodar").

| Modelo | Origem |
|---|---|
| TinyBERT pré-treinado `huawei-noah/TinyBERT_General_4L_312D` | Baixado automaticamente do Hugging Face Hub na 1ª execução (cache do `transformers`, fora do repositório) |
| llama3.1:latest (8B) para a geração sintética | Servido pelo Ollama local (`ollama pull llama3.1:latest`), fora do repositório |
| TextCNN spaCy | Arquitetura definida em `src/textcnn.py`; treinada do zero em cada run |
| `results/_work/` | Diretório temporário do HF `Trainer` (vazio, ignorado pelo git) |

### 4. Resultados finais (`results/`)

36 runs: 4 cenários (10/25/50/75%) × 3 braços × 3 seeds, todos na mesma máquina.

**Tabelas e relatórios**

| Arquivo | Conteúdo |
|---|---|
| `results/runs.csv` | Uma linha por run (36): F1 macro, acurácia, precisão, recall, tempos, FLOPs, memória, latência, nº de amostras reais/sintéticas, features selecionadas da CNN |
| `results/predictions/<braço>__<ratio>__<seed>.csv` | 36 arquivos com rótulo verdadeiro, predição e probabilidades por classe para cada frase do teste (base do bootstrap pareado) |
| `results/summary.csv` | Média ± dp por (braço, ratio) |
| `results/contrasts.csv` | ΔF1 com IC 95%, Δcusto, razão de custo, custo por p.p. e veredito, por contraste/ratio/métrica de custo |
| `results/efficiency.csv` | Ganho de F1 por PFLOP e por minuto extra em relação ao TinyBERT escasso |
| `results/report.md` | Relatório completo gerado automaticamente, que responde à pergunta de pesquisa |
| `results/resumo.md` | Resumo dos achados (tabela principal e conclusões), escrito para o TCC |

**Figuras**

| Arquivo | Conteúdo |
|---|---|
| `results/fig_f1_vs_cost_total_time_s.png` | F1 × tempo total (log) com fronteira de Pareto, um painel por cenário; linha de referência do tempo de anotação humana |
| `results/fig_f1_vs_cost_total_flops.png`, `results/fig_f1_vs_cost_train_time_s.png`, `results/fig_f1_vs_cost_train_flops.png` | Idem, para FLOPs totais e para custo só de treino |
| `results/fig_f1_by_ratio.png` | F1 macro de cada braço por nível de escassez |
| `results/fig_cost_breakdown.png` | Composição dos FLOPs (fine-tuning, TextCNN, geração LLM) |
| `results/fig_cost_by_ratio.png` | Tempo total e FLOPs de cada braço ao longo da escassez, com a série de anotação humana no painel de tempo |
| `results/fig_delta_f1_forest.png` | ΔF1 pareado com IC 95% de todos os contrastes |
| `results/fig_delta_f1_by_ratio.png` | Ganho de TL e DA sobre o TinyBERT escasso e confronto TL − DA por escassez |
| `results/fig_ce_plane.png` | Plano custo-efetividade TL vs DA (ΔF1 × razão de custo) |
| `results/fig_time_f1_human.png` | Tempo total e F1 lado a lado por cenário, incluindo a anotação humana (títulos removidos × 4,6 s + treino) como referência de resultado ótimo |

### 5. Resultados de desenhos anteriores (`results_archive/`)

Mantidos por rastreabilidade; **não entram nas conclusões**.

| Pasta | Conteúdo |
|---|---|
| `results_archive/10pct_parcial_195features/` | Run parcial de 10% (seed 42) do desenho antigo: híbrido usando as 195 features da CNN sem seleção, e braço `textcnn_restricted`. `runs.csv` + `predictions/` |
| `results_archive/10pct_da_csv_baixado/` | Braço DA de 10% treinado com os CSVs sintéticos baixados da origem, em vez de geração ao vivo. `runs.csv` + `predictions/` |

### 6. Logs de execução (`logs/`)

Saída completa (stdout/stderr) do `run_experiment.py` em cada execução, com o
andamento dos runs e as métricas de treino do HF `Trainer`.

| Arquivo | Execução |
|---|---|
| `logs/run_25_50_75pct.log` | Cenários 25, 50 e 75% (27 runs) — **resultados finais** |
| `logs/run_10pct.log` | Cenário 10%, braços TinyBERT escasso e TL — **resultados finais** |
| `logs/run_10pct_da_live.log` | Cenário 10%, braço DA com geração ao vivo (3 runs) — **resultados finais** |
| `logs/run_full.log`, `logs/run_50pct.log` | Tentativas anteriores, interrompidas ou de desenhos superados (ver `results_archive/`) |

### 7. Gerados localmente e fora do material entregue

| Caminho | Motivo |
|---|---|
| `results_smoke/` | Saída do teste rápido (`--smoke`, subamostra e 1 época); só valida o pipeline. Ignorado pelo git |
| `results/_work/` | Diretório temporário do treino (vazio). Ignorado pelo git |
| `.venv/`, `__pycache__/` | Ambiente virtual e cache do Python; recriados com `uv sync` |


## Como rodar

```bash
uv sync

# 1. teste rápido do pipeline (subamostra, 1 época) → results_smoke/
uv run python run_experiment.py --smoke

# 2. cenários de RATIOS em config.py (10, 25, 50, 75) → results/
uv run python run_experiment.py
uv run python run_experiment.py --ratios 25        # só um cenário

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

Execuções longas (cerca de 4 h para 25/50/75%) foram lançadas em segundo plano, com o log em `logs/`:

```bash
nohup caffeinate -i .venv/bin/python run_experiment.py --ratios 25 50 75 > logs/run_25_50_75pct.log 2>&1 &
uv run python analyze.py   # regenera tabelas, figuras e report.md a partir de results/runs.csv
```

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
