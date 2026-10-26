# Resumo dos achados — TL (TextCNN → TinyBERT) vs TinyBERT + DA

36 runs: 4 níveis de escassez (10/25/50/75% dos dados reais removidos) × 3 braços × 3 seeds,
mesmo test set (690 frases), mesma máquina (TinyBERT em MPS, TextCNN em CPU, llama3.1 8B via Ollama).
Números detalhados em `report.md`; ΔF1 com IC 95% por bootstrap pareado.

## F1 macro (média de 3 seeds) e custo total (treino + geração)

| removido | TinyBERT escasso | TL (TextCNN→TinyBERT) | TinyBERT + DA | ΔF1 TL − DA [IC 95%] | tempo TL ÷ DA | FLOPs TL ÷ DA |
|---|---|---|---|---|---|---|
| 10% | 0.888 · 0.5 min | 0.897 · 5.9 min | 0.897 · 5.6 min | −0.02 p.p. [−1.07; +1.04] | 1.06× | 0.071× |
| 25% | 0.897 · 0.5 min | 0.891 · 5.2 min | 0.886 · 12.5 min | +0.57 p.p. [−0.74; +1.92] | 0.42× | 0.025× |
| 50% | 0.870 · 0.3 min | 0.869 · 3.6 min | 0.876 · 24.0 min | −0.63 p.p. [−2.19; +0.98] | 0.15× | 0.008× |
| 75% | 0.672 · 0.2 min | 0.813 · 1.8 min | 0.864 · 35.5 min | **−5.05 p.p. [−7.53; −2.62]** | 0.052× | 0.003× |

## Achados

1. **Até 50% de escassez, TL e DA empatam em qualidade.** Nenhuma diferença TL − DA é
   significativa em 10, 25 e 50% (todos os IC cruzam zero). Wilcoxon sobre os 12 pares
   (ratio, seed): mediana −0,77 p.p., p = 0,18.
2. **Quando o custo da geração entra na conta, o TL é muito mais barato, e a vantagem cresce
   com a escassez.** O custo do DA cresce linearmente com o número de frases sintéticas
   (~1 s/frase no llama3.1). O do TL cai, porque há menos dados para treinar. TL ÷ DA em tempo
   total: 1,06× → 0,42× → 0,15× → 0,05×. Em FLOPs, o TL é de 14× a 360× mais barato.
3. **Veredito por cenário (margem de não-inferioridade de 1 p.p.):**
   - **25%:** TL **custo-efetivo**, porque é não-inferior e 2,4× mais rápido (41× menos FLOPs).
   - **50%:** TL **inconclusivo**. É 6,6× mais rápido, mas o IC (−2,2 p.p.) não demonstra
     não-inferioridade.
   - **75%:** **trade-off**. O DA é 5 p.p. melhor, mas custa +33,7 min (~6,7 min por p.p.) e
     +4,2 PFLOP a mais que o TL.
   - **10%:** **não compensa**. O custo é praticamente igual em tempo, e nenhum dos dois ganha
     do TinyBERT escasso.
4. **Com até 50% removidos, nenhuma das estratégias supera o TinyBERT escasso**, que é a opção
   mais barata (≤ 0,5 min). O DA em 25% chega a ser significativamente pior (−1,09 p.p.).
   O custo extra só se paga em escassez extrema.
5. **Em 75%, as duas estratégias resgatam o modelo.** O TinyBERT escasso fica instável
   (dp = 0,156). No seed 42 a classe negativa colapsa (F1 0,09, só 84 exemplos negativos).
   - **TL:** +14,1 p.p. sobre o escasso, com dp 0,012 e F1 da classe negativa de 0,85–0,87 em
     todos os seeds. Recupera 74% do ganho do DA com ~5% do tempo extra: 8,3 p.p. por minuto extra,
     contra 0,54 do DA.
   - **DA:** +19,1 p.p. A vantagem sobre o TL vem principalmente da classe positiva
     (F1 0,79 vs 0,63–0,71).
6. **Se os dados sintéticos já existirem (só treino), a vantagem do TL some em tempo.** A TextCNN
   com cross-fitting de 5 folds em CPU deixa o treino do TL 3–9,5× mais lento que o do DA. Os
   FLOPs ficam parecidos (0,3–1,1×). Na inferência, o TL tem cerca de 2× a latência
   (~5 ms vs ~2,3 ms por frase) e +6,5% de parâmetros.

## Resposta à pergunta de pesquisa

**Sim, mas com condições.** Considerando o pipeline completo (inclusive a geração dos dados
sintéticos), o transfer learning TextCNN → TinyBERT tem custo-benefício superior ao do
TinyBERT + DA em escassez moderada. Em 25–50% removidos, entrega F1 estatisticamente
indistinguível com custo de 2,4× a 6,6× menor em tempo e de 40× a 120× menor em FLOPs.

Em escassez extrema (75%) o TL continua sendo a forma mais eficiente de evitar o colapso do
TinyBERT. Se a meta for o maior F1 absoluto, porém, o DA vence por 5 p.p., a um custo cerca
de 19× maior.

Com pouca escassez (≤ 10%), nenhuma das duas estratégias compensa frente ao TinyBERT treinado
só com os dados disponíveis.

## Limitações

- 3 seeds e 690 frases de teste: diferenças menores que ~1–2 p.p. ficam dentro do ruído. Exemplo:
  o TinyBERT escasso teve F1 maior em 25% do que em 10%.
- Os tempos valem para esta máquina. Com GPU dedicada ou servidor de inferência, a geração
  do LLM ficaria mais barata e a vantagem do TL em tempo total diminuiria. Os FLOPs servem de
  referência independente de hardware, mas são estimativas analíticas.
- Os runs de 10% são de 28/09 e os de 25/50/75% de 24–25/10, na mesma máquina. A velocidade
  de geração ficou consistente (1,01–1,08 s/frase em todos os cenários).
