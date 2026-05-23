# Resultados - Texto-base

## 1. Seleção e estruturação do dataset

Foram carregados 21.799 registros do PTB-XL. Com a estratégia multiclasse `strict_single_label`, foram mantidos 16.244 registros com exatamente uma superclass diagnóstica entre `NORM`, `MI`, `STTC`, `CD` e `HYP`. Foram removidos 411 registros sem superclass diagnóstica considerada e 5.144 registros multi-label.

A distribuição final ficou desbalanceada: `NORM` com 9.069 registros, `MI` com 2.532, `STTC` com 2.400, `CD` com 1.708 e `HYP` com 535 registros.

## 2. Pré-processamento e extração de características

Foram extraídas características estatísticas simples por derivação dos sinais em `records100`, incluindo média, desvio padrão, mínimo, máximo, mediana, percentis, amplitude pico-a-pico, energia, RMS, assimetria e curtose. Também foram incluídos metadados simples como sexo, `age_clean` e `age_is_anon_90_plus`.

Valores `age == 300` foram tratados como anonimização de pacientes com idade real maior ou igual a 90 anos, não como idade fisiológica. A imputação de `age_clean` foi mantida dentro dos pipelines de treino.

## 3. Balanceamento de classes

Foram comparados cenários sem SMOTE e com SMOTE. O balanceamento foi aplicado apenas ao conjunto de treino, preservando validação e teste com a distribuição observada.

Os novos artefatos de estratificação mostram que o desbalanceamento permaneceu nos três splits, com predominância da classe NORM e menor representação da classe HYP. Essa preservação é desejável para validação e teste, pois evita avaliar os modelos em uma distribuição artificial. As estratégias de balanceamento foram restritas ao treino: SMOTE nos modelos clássicos e, nos modelos de deep learning, class weights, focal loss e WeightedRandomSampler quando configurados. Como o SMOTE não melhorou uniformemente todos os modelos, a avaliação priorizou F1 macro, balanced accuracy e métricas por classe, além da acurácia.

## 4. Treinamento e comparação dos modelos

Foram avaliados Regressão Logística, SVM linear, Random Forest, LightGBM e CatBoost. A seleção leve de hiperparâmetros utilizou o fold 9, e a avaliação final foi feita no fold 10.

## 5. Avaliação de desempenho

O melhor modelo por F1 macro no teste foi `lightgbm_without_smote`, com F1 macro de 0,5610 e acurácia de 0,6727. O melhor modelo por acurácia foi `lightgbm_with_smote`, com acurácia de 0,6770 e F1 macro de 0,5590.

Nenhum modelo clássico atingiu a meta de 80% de acurácia nesta execução com features estatísticas. Esse resultado motivou a inclusão de uma arquitetura ResNet1D leve, que usa o sinal bruto do ECG em vez de apenas estatísticas agregadas.

## 6. Interpretabilidade

Foram gerados ranking global de features e visualizações SHAP para o melhor modelo de árvore compatível, além de exemplos locais com LIME. As explicações devem ser interpretadas como importância para o comportamento do modelo, não como causalidade ou interpretação clínica definitiva.

Enquanto o SHAP foi utilizado para observar padrões globais de importância das variáveis, o LIME foi empregado para examinar predições individuais. Essa distinção é relevante porque a interpretabilidade global indica quais atributos tendem a influenciar o comportamento geral do modelo, enquanto a interpretabilidade local permite analisar por que uma amostra específica recebeu determinada classificação. Nos exemplos gerados, as barras positivas indicam atributos que favoreceram a classe predita, enquanto as barras negativas indicam atributos que reduziram a evidência para essa classe. Essas explicações contribuem para transparência do modelo, mas não indicam causalidade médica nem validação clínica.

Bloco LaTeX sugerido para a subseção de interpretabilidade:

```latex
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.82\linewidth]{figuras_tcc/resultados/lime_exemplos_locais.pdf}
    \caption{Exemplos de explicações locais geradas com LIME para predições individuais.}
    \label{fig:lime-exemplos-locais}
\end{figure}
```

## 7. Identificação da abordagem mais adequada

Considerando F1 macro como métrica principal, a melhor abordagem nesta execução foi LightGBM sem SMOTE. Considerando acurácia, LightGBM com SMOTE apresentou o maior valor, mas a diferença em relação ao cenário sem SMOTE foi pequena.

Como melhoria futura, recomenda-se revisar a estratégia de rótulos, enriquecer a extração de características, ajustar a arquitetura CNN residual de forma controlada e considerar formulação multi-label em trabalhos futuros.

## ResNet1D leve

Foi adicionada uma ResNet1D leve em PyTorch para classificação multiclasse a partir dos sinais brutos em `records500`. A rede utiliza normalização por canal calculada apenas no treino, class weights para lidar com desbalanceamento, `EarlyStopping`, `ReduceLROnPlateau` e checkpoint do melhor modelo por `val_loss`.

Na execução realizada, a ResNet1D leve obteve acurácia de 0,7503 e F1 macro de 0,6698 no fold de teste. O resultado superou os modelos clássicos, mas ainda não atingiu a meta de 80% de acurácia. Esse achado deve ser documentado honestamente, sem afirmar utilidade clínica definitiva.

## Baseline CNN simples

Após a configuração de uma `.venv` local com TensorFlow, o baseline CNN simples foi executado com `records100`. Na execução realizada, o modelo obteve acurácia de 0,7570 e F1 macro de 0,5768 no fold de teste.

Esse baseline apresentou acurácia maior que os modelos clássicos, mas F1 macro inferior à ResNet1D leve, indicando que ainda houve dificuldade em equilibrar desempenho entre as classes.

## InceptionTime 1D forte

Foi executada uma arquitetura InceptionTime 1D mais forte em PyTorch usando `records500`, kernels temporais múltiplos, conexões residuais, focal loss com alpha por classe e aumentações leves apenas no treino. O melhor checkpoint foi escolhido pelo F1 macro de validação, mantendo o fold de teste reservado para a avaliação final.

Na execução em CPU, o modelo obteve acurácia de 0,7824 e F1 macro de 0,6840 no fold de teste. O resultado melhorou a acurácia em relação à ResNet1D leve, mas não superou a meta de 80%. A F1 macro ficou próxima da ResNet1D leve, indicando que ainda há limitação de desempenho entre classes minoritárias.

## Deep learning pesado e desbalanceamento

Foi adicionado um experimento de deep learning mais robusto em PyTorch usando records500 e sinais brutos. As estrategias de desbalanceamento foram aplicadas somente no treino, incluindo pesos por classe, focal loss, WeightedRandomSampler e aumentacoes leves. Validacao e teste foram preservados sem balanceamento artificial.

No conjunto de teste, o modelo deep_learning_heavy_resnet1d_se obteve acuracia de 0.7248, F1 macro de 0.6390, balanced accuracy de 0.6677 e recall macro de 0.6677. O modelo nao atingiu 80% de acuracia. Esses resultados devem ser interpretados como avaliacao experimental, sem validacao clinica definitiva.
