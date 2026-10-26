# Transfer learning (TextCNN → TinyBERT) vs TinyBERT + data augmentation

**Pergunta:** transfer learning oferece custo-benefício (performance / custo computacional) para classificação de texto financeiro com dados escassos?

Contraste principal: `hybrid_restricted` (TL, só dados reais) vs `tinybert_augmented` (TinyBERT + DA com 2071 frases sintéticas no máximo). IC 95% por bootstrap pareado nas frases de teste, média entre seeds. Margem de não-inferioridade: 1.0 p.p. de F1 macro.

## Veredito por nível de escassez

| ratio | cost_metric | ΔF1 (p.p.) | IC inf | IC sup | p_bootstrap | custo TL / custo DA | Δcusto por p.p. | verdict |
|---|---|---|---|---|---|---|---|---|
| 10 | total_time_s | -0.02353 | -1.068 | 1.036 | 0.992 | 1.058 | -823.7 | não compensa (sem ganho significativo e mais caro) |
| 10 | total_flops | -0.02353 | -1.068 | 1.036 | 0.992 | 0.07055 | 2.377e+16 | inconclusivo (mais barato; não-inferioridade não demonstrada) |
| 10 | train_time_s | -0.02353 | -1.068 | 1.036 | 0.992 | 9.529 | -1.341e+04 | não compensa (sem ganho significativo e mais caro) |
| 10 | train_flops | -0.02353 | -1.068 | 1.036 | 0.992 | 1.1 | -1.646e+14 | não compensa (sem ganho significativo e mais caro) |
| 25 | total_time_s | 0.5653 | -0.7403 | 1.916 | 0.384 | 0.4179 | -772.2 | custo-efetivo (não-inferior e mais barato) |
| 25 | total_flops | 0.5653 | -0.7403 | 1.916 | 0.384 | 0.02451 | -2.495e+15 | custo-efetivo (não-inferior e mais barato) |
| 25 | train_time_s | 0.5653 | -0.7403 | 1.916 | 0.384 | 8.609 | 489.9 | não compensa (sem ganho significativo e mais caro) |
| 25 | train_flops | 0.5653 | -0.7403 | 1.916 | 0.384 | 0.9176 | -5.628e+12 | custo-efetivo (não-inferior e mais barato) |
| 50 | total_time_s | -0.6284 | -2.19 | 0.98 | 0.488 | 0.152 | 1942 | inconclusivo (mais barato; não-inferioridade não demonstrada) |
| 50 | total_flops | -0.6284 | -2.19 | 0.98 | 0.488 | 0.008318 | 4.487e+15 | inconclusivo (mais barato; não-inferioridade não demonstrada) |
| 50 | train_time_s | -0.6284 | -2.19 | 0.98 | 0.488 | 6.08 | -290.7 | não compensa (sem ganho significativo e mais caro) |
| 50 | train_flops | -0.6284 | -2.19 | 0.98 | 0.488 | 0.6131 | 2.375e+13 | inconclusivo (mais barato; não-inferioridade não demonstrada) |
| 75 | total_time_s | -5.053 | -7.529 | -2.62 | 0.001 | 0.05196 | 400 | trade-off (mais barato, porém inferior) |
| 75 | total_flops | -5.053 | -7.529 | -2.62 | 0.001 | 0.002781 | 8.388e+14 | trade-off (mais barato, porém inferior) |
| 75 | train_time_s | -5.053 | -7.529 | -2.62 | 0.001 | 3.066 | -14.77 | dominado (pior e mais caro) |
| 75 | train_flops | -5.053 | -7.529 | -2.62 | 0.001 | 0.3062 | 5.301e+12 | trade-off (mais barato, porém inferior) |

Wilcoxon pareado sobre todos os (ratio, seed): n=12, mediana ΔF1=-0.77 p.p., p=0.1763

## Contrastes secundários

| contrast | ratio | delta_f1 | ci_low | ci_high | cost_ratio | verdict |
|---|---|---|---|---|---|---|
| hybrid_restricted vs tinybert_restricted | 10 | 0.9131 | -0.2595 | 2.116 | 11.41 | não compensa (sem ganho significativo e mais caro) |
| hybrid_restricted vs tinybert_restricted | 25 | -0.5223 | -1.538 | 0.4468 | 11.59 | não compensa (sem ganho significativo e mais caro) |
| hybrid_restricted vs tinybert_restricted | 50 | -0.09263 | -1.301 | 1.059 | 12.09 | não compensa (sem ganho significativo e mais caro) |
| hybrid_restricted vs tinybert_restricted | 75 | 14.06 | 11.71 | 16.43 | 12.03 | trade-off (melhor, porém mais caro) |
| tinybert_augmented vs tinybert_restricted | 10 | 0.9366 | -0.121 | 1.967 | 10.78 | não compensa (sem ganho significativo e mais caro) |
| tinybert_augmented vs tinybert_restricted | 25 | -1.088 | -2.073 | -0.1147 | 27.74 | dominado (pior e mais caro) |
| tinybert_augmented vs tinybert_restricted | 50 | 0.5358 | -1.381 | 2.428 | 79.55 | não compensa (sem ganho significativo e mais caro) |
| tinybert_augmented vs tinybert_restricted | 75 | 19.11 | 16.3 | 22 | 231.5 | trade-off (melhor, porém mais caro) |

## Eficiência vs TinyBERT escasso (ganho de F1 por custo extra)

| arm | ratio | f1_gain_pp | extra_pflop | extra_minutes | pp_per_pflop | pp_per_minute | dominates_reference_flops |
|---|---|---|---|---|---|---|---|
| hybrid_restricted | 10 | 0.9131 | 0.007749 | 5.36 | 117.8 | 0.1704 | False |
| hybrid_restricted | 25 | -0.5223 | 0.006498 | 4.772 | -80.37 | -0.1095 | False |
| hybrid_restricted | 50 | -0.09263 | 0.004379 | 3.343 | -21.15 | -0.02771 | False |
| hybrid_restricted | 75 | 14.06 | 0.002183 | 1.693 | 6441 | 8.306 | False |
| tinybert_augmented | 10 | 0.9366 | 0.5672 | 5.037 | 1.651 | 0.186 | False |
| tinybert_augmented | 25 | -1.088 | 1.417 | 12.05 | -0.7676 | -0.09028 | False |
| tinybert_augmented | 50 | 0.5358 | 2.824 | 23.68 | 0.1897 | 0.02263 | False |
| tinybert_augmented | 75 | 19.11 | 4.241 | 35.38 | 4.507 | 0.5403 | False |

## Resumo por braço

| arm | ratio | n | f1_macro | f1_std | accuracy | generation_time_s | train_time_s | total_time_s | total_flops | latency_ms | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_restricted | 10 | 3 | 0.8967 | 0.00807 | 0.915 | 0 | 352.5 | 352.5 | 4.247e+13 | 4.869 | 1.528e+07 |
| hybrid_restricted | 25 | 3 | 0.8913 | 0.01072 | 0.9111 | 0 | 313.3 | 313.3 | 3.544e+13 | 4.964 | 1.528e+07 |
| hybrid_restricted | 50 | 3 | 0.8693 | 0.01007 | 0.8932 | 0 | 218.6 | 218.6 | 2.365e+13 | 5.114 | 1.528e+07 |
| hybrid_restricted | 75 | 3 | 0.813 | 0.0122 | 0.8507 | 0 | 110.8 | 110.8 | 1.182e+13 | 5.093 | 1.528e+07 |
| tinybert_augmented | 10 | 3 | 0.8969 | 0.01053 | 0.9155 | 296.1 | 36.99 | 333.1 | 6.020e+14 | 2.453 | 1.435e+07 |
| tinybert_augmented | 25 | 3 | 0.8856 | 0.002442 | 0.9058 | 713.4 | 36.4 | 749.8 | 1.446e+15 | 2.333 | 1.435e+07 |
| tinybert_augmented | 50 | 3 | 0.8756 | 0.009367 | 0.8981 | 1403 | 35.96 | 1439 | 2.844e+15 | 2.342 | 1.435e+07 |
| tinybert_augmented | 75 | 3 | 0.8635 | 0.005817 | 0.8826 | 2096 | 36.13 | 2132 | 4.251e+15 | 2.338 | 1.435e+07 |
| tinybert_restricted | 10 | 3 | 0.8875 | 0.006559 | 0.9097 | 0 | 30.9 | 30.9 | 3.472e+13 | 2.284 | 1.435e+07 |
| tinybert_restricted | 25 | 3 | 0.8965 | 0.009487 | 0.9155 | 0 | 27.03 | 27.03 | 2.894e+13 | 2.33 | 1.435e+07 |
| tinybert_restricted | 50 | 3 | 0.8702 | 0.004162 | 0.8932 | 0 | 18.09 | 18.09 | 1.928e+13 | 2.316 | 1.435e+07 |
| tinybert_restricted | 75 | 3 | 0.6724 | 0.1561 | 0.7986 | 0 | 9.209 | 9.209 | 9.638e+12 | 2.309 | 1.435e+07 |

## Features da TextCNN usadas pelo TinyBERT (top-12 por informação mútua)

Por run (ordem decrescente de importância):

- `hybrid_restricted` 10% seed 42: prob_neutral, prob_positive, prob_negative, mean_16, mean_51, mean_32, mean_43, mean_22, mean_85, max_31, mean_69, max_10
- `hybrid_restricted` 10% seed 43: prob_neutral, prob_positive, prob_negative, mean_73, mean_51, mean_87, mean_79, mean_91, mean_71, mean_66, max_87, max_73
- `hybrid_restricted` 10% seed 44: prob_neutral, prob_positive, prob_negative, mean_14, mean_28, mean_20, mean_51, mean_73, mean_71, max_20, max_51, mean_3
- `hybrid_restricted` 25% seed 42: prob_neutral, prob_positive, mean_13, prob_negative, mean_87, mean_49, mean_41, mean_32, mean_52, max_13, mean_9, mean_26
- `hybrid_restricted` 50% seed 42: prob_neutral, prob_positive, prob_negative, mean_29, mean_24, mean_34, mean_30, mean_53, mean_76, mean_73, mean_1, mean_13
- `hybrid_restricted` 75% seed 42: prob_neutral, prob_positive, mean_52, prob_negative, mean_51, mean_73, mean_6, mean_28, mean_5, mean_85, mean_87, mean_74
- `hybrid_restricted` 25% seed 43: prob_neutral, prob_positive, mean_11, prob_negative, mean_20, mean_7, mean_95, mean_58, mean_23, mean_55, mean_63, mean_73
- `hybrid_restricted` 50% seed 43: prob_neutral, prob_positive, prob_negative, mean_63, mean_78, mean_75, mean_10, mean_59, mean_33, mean_47, mean_91, mean_13
- `hybrid_restricted` 75% seed 43: prob_neutral, prob_positive, mean_39, mean_62, mean_41, mean_94, prob_negative, mean_74, mean_58, mean_26, mean_57, mean_53
- `hybrid_restricted` 25% seed 44: prob_neutral, prob_positive, mean_13, mean_38, mean_60, mean_73, prob_negative, mean_32, mean_49, mean_87, max_13, mean_91
- `hybrid_restricted` 50% seed 44: prob_neutral, prob_positive, prob_negative, mean_82, max_5, mean_5, mean_2, mean_87, mean_23, mean_38, max_87, mean_59
- `hybrid_restricted` 75% seed 44: prob_neutral, prob_positive, prob_negative, mean_93, mean_49, mean_89, mean_74, mean_31, mean_71, mean_38, mean_73, mean_20

Frequência entre runs:

| feature | n_seeds_selecionada |
|---|---|
| prob_neutral | 12 |
| prob_positive | 12 |
| prob_negative | 12 |
| mean_73 | 7 |
| mean_87 | 5 |
| mean_51 | 4 |
| mean_13 | 4 |
| mean_32 | 3 |
| mean_91 | 3 |
| mean_71 | 3 |
| mean_20 | 3 |
| mean_49 | 3 |
| mean_74 | 3 |
| mean_38 | 3 |
| mean_85 | 2 |
| max_87 | 2 |
| mean_28 | 2 |
| mean_41 | 2 |
| mean_52 | 2 |
| max_13 | 2 |
| mean_26 | 2 |
| mean_53 | 2 |
| mean_5 | 2 |
| mean_58 | 2 |
| mean_23 | 2 |
| mean_63 | 2 |
| mean_59 | 2 |
| mean_16 | 1 |
| mean_43 | 1 |
| mean_22 | 1 |
| max_31 | 1 |
| mean_69 | 1 |
| max_10 | 1 |
| mean_79 | 1 |
| mean_66 | 1 |
| max_73 | 1 |
| mean_14 | 1 |
| max_20 | 1 |
| max_51 | 1 |
| mean_3 | 1 |
| mean_9 | 1 |
| mean_29 | 1 |
| mean_24 | 1 |
| mean_34 | 1 |
| mean_30 | 1 |
| mean_76 | 1 |
| mean_1 | 1 |
| mean_6 | 1 |
| mean_11 | 1 |
| mean_7 | 1 |
| mean_95 | 1 |
| mean_55 | 1 |
| mean_78 | 1 |
| mean_75 | 1 |
| mean_10 | 1 |
| mean_33 | 1 |
| mean_47 | 1 |
| mean_39 | 1 |
| mean_62 | 1 |
| mean_94 | 1 |
| mean_57 | 1 |
| mean_60 | 1 |
| mean_82 | 1 |
| max_5 | 1 |
| mean_2 | 1 |
| mean_93 | 1 |
| mean_89 | 1 |
| mean_31 | 1 |

## Figuras

- `fig_f1_vs_cost_<custo>.png` — performance × custo com fronteira de Pareto (total_* inclui a geração LLM; train_* considera só o treino; total_time_s requer tempo de geração medido)
- `fig_f1_by_ratio.png` — F1 por nível de escassez
- `fig_cost_breakdown.png` — composição do custo
- `fig_delta_f1_forest.png` — ΔF1 pareado com IC 95%
- `fig_ce_plane.png` — plano custo-efetividade TL vs DA (ΔF1 × razão de custo), um ponto por escassez
- `fig_cost_by_ratio.png` — custo total (tempo e FLOPs) de cada braço ao longo da escassez
- `fig_delta_f1_by_ratio.png` — ganho de TL e DA sobre o TinyBERT escasso e confronto TL − DA por escassez
