# Transfer learning (TextCNN → TinyBERT) vs TinyBERT + data augmentation

**Pergunta:** transfer learning oferece custo-benefício (performance / custo computacional) para classificação de texto financeiro com dados escassos?

Contraste principal: `hybrid_restricted` (TL, só dados reais) vs `tinybert_augmented` (TinyBERT + DA com 277 frases sintéticas no máximo). IC 95% por bootstrap pareado nas frases de teste, média entre seeds. Margem de não-inferioridade: 1.0 p.p. de F1 macro.

## Veredito por nível de escassez

| ratio | cost_metric | ΔF1 (p.p.) | IC inf | IC sup | p_bootstrap | custo TL / custo DA | Δcusto por p.p. | verdict |
|---|---|---|---|---|---|---|---|---|
| 10 | total_time_s | -0.02353 | -1.068 | 1.036 | 0.992 | 1.058 | -823.7 | não compensa (sem ganho significativo e mais caro) |
| 10 | total_flops | -0.02353 | -1.068 | 1.036 | 0.992 | 0.07055 | 2.377e+16 | inconclusivo (mais barato; não-inferioridade não demonstrada) |
| 10 | train_time_s | -0.02353 | -1.068 | 1.036 | 0.992 | 9.529 | -1.341e+04 | não compensa (sem ganho significativo e mais caro) |
| 10 | train_flops | -0.02353 | -1.068 | 1.036 | 0.992 | 1.1 | -1.646e+14 | não compensa (sem ganho significativo e mais caro) |

Wilcoxon pareado sobre todos os (ratio, seed): n=3, mediana ΔF1=0.56 p.p., p=1

## Contrastes secundários

| contrast | ratio | delta_f1 | ci_low | ci_high | cost_ratio | verdict |
|---|---|---|---|---|---|---|
| hybrid_restricted vs tinybert_restricted | 10 | 0.9131 | -0.2595 | 2.116 | 11.41 | não compensa (sem ganho significativo e mais caro) |
| tinybert_augmented vs tinybert_restricted | 10 | 0.9366 | -0.121 | 1.967 | 10.78 | não compensa (sem ganho significativo e mais caro) |

## Eficiência vs TinyBERT escasso (ganho de F1 por custo extra)

| arm | ratio | f1_gain_pp | extra_pflop | extra_minutes | pp_per_pflop | pp_per_minute | dominates_reference_flops |
|---|---|---|---|---|---|---|---|
| hybrid_restricted | 10 | 0.9131 | 0.007749 | 5.36 | 117.8 | 0.1704 | False |
| tinybert_augmented | 10 | 0.9366 | 0.5672 | 5.037 | 1.651 | 0.186 | False |

## Resumo por braço

| arm | ratio | n | f1_macro | f1_std | accuracy | generation_time_s | train_time_s | total_time_s | total_flops | latency_ms | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_restricted | 10 | 3 | 0.8967 | 0.00807 | 0.915 | 0 | 352.5 | 352.5 | 4.247e+13 | 4.869 | 1.528e+07 |
| tinybert_augmented | 10 | 3 | 0.8969 | 0.01053 | 0.9155 | 296.1 | 36.99 | 333.1 | 6.020e+14 | 2.453 | 1.435e+07 |
| tinybert_restricted | 10 | 3 | 0.8875 | 0.006559 | 0.9097 | 0 | 30.9 | 30.9 | 3.472e+13 | 2.284 | 1.435e+07 |

## Features da TextCNN usadas pelo TinyBERT (top-12 por informação mútua)

Por run (ordem decrescente de importância):

- `hybrid_restricted` 10% seed 42: prob_neutral, prob_positive, prob_negative, mean_16, mean_51, mean_32, mean_43, mean_22, mean_85, max_31, mean_69, max_10
- `hybrid_restricted` 10% seed 43: prob_neutral, prob_positive, prob_negative, mean_73, mean_51, mean_87, mean_79, mean_91, mean_71, mean_66, max_87, max_73
- `hybrid_restricted` 10% seed 44: prob_neutral, prob_positive, prob_negative, mean_14, mean_28, mean_20, mean_51, mean_73, mean_71, max_20, max_51, mean_3

Frequência entre runs:

| feature | n_seeds_selecionada |
|---|---|
| prob_neutral | 3 |
| prob_positive | 3 |
| prob_negative | 3 |
| mean_51 | 3 |
| mean_73 | 2 |
| mean_71 | 2 |
| mean_16 | 1 |
| mean_32 | 1 |
| mean_43 | 1 |
| mean_22 | 1 |
| mean_85 | 1 |
| max_31 | 1 |
| mean_69 | 1 |
| max_10 | 1 |
| mean_87 | 1 |
| mean_79 | 1 |
| mean_91 | 1 |
| mean_66 | 1 |
| max_87 | 1 |
| max_73 | 1 |
| mean_14 | 1 |
| mean_28 | 1 |
| mean_20 | 1 |
| max_20 | 1 |
| max_51 | 1 |
| mean_3 | 1 |

## Figuras

- `fig_f1_vs_cost_<custo>.png` — performance × custo com fronteira de Pareto (total_* inclui a geração LLM; train_* considera só o treino; total_time_s requer tempo de geração medido)
- `fig_f1_by_ratio.png` — F1 por nível de escassez
- `fig_cost_breakdown.png` — composição do custo
- `fig_delta_f1_forest.png` — ΔF1 pareado com IC 95%
