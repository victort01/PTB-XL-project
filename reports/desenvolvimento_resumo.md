# Desenvolvimento - Texto-base

## Visão geral da solução

A solução foi organizada como um pipeline reprodutível para classificação multiclasse de ECGs de 12 derivações do PTB-XL. O projeto separa a narrativa experimental nos notebooks e a lógica reutilizável em módulos Python, favorecendo rastreabilidade, testes e reutilização.

## Arquitetura do projeto

A estrutura segue uma adaptação enxuta do padrão cookiecutter para projetos de dados. Os dados permanecem em `data/`, os notebooks em `notebooks/`, as funções reutilizáveis em `src/tcc_ecg/`, os artefatos acadêmicos em `reports/` e os modelos treinados em `models/`.

## Fluxo dos dados

O fluxo inicia com a leitura de `ptbxl_database.csv` e `scp_statements.csv`, seguida pelo tratamento explícito da idade anonimizada, construção dos rótulos diagnósticos, extração de características estatísticas dos sinais e divisão pelos folds oficiais do PTB-XL. O balanceamento por SMOTE é aplicado somente no treino.

## Componentes principais

Os módulos principais incluem carregamento de dados, construção de rótulos, extração de características, preprocessamento, balanceamento, treinamento de modelos, deep learning, avaliação e interpretabilidade. Essa divisão reduz duplicação nos notebooks e facilita a verificação de cada etapa.

## Tecnologias utilizadas

O projeto utiliza Python, pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, WFDB, Matplotlib, Seaborn, LightGBM, CatBoost, SHAP e LIME. O baseline simples de deep learning pode ser executado com TensorFlow, enquanto a ResNet1D leve usa PyTorch.

## Justificativa das escolhas técnicas

As características estatísticas por derivação foram escolhidas por serem simples, reproduzíveis e adequadas como baseline para modelos tradicionais. A F1-score macro foi priorizada devido ao desbalanceamento entre classes. Como os modelos clássicos ficaram abaixo da meta de 80% de acurácia, foram adicionadas redes 1D sobre o sinal bruto do ECG. A ResNet1D leve funciona como baseline neural residual; a InceptionTime 1D forte usa kernels temporais múltiplos para capturar padrões em diferentes escalas. Ambas usam normalização calculada apenas no treino, class weights, checkpoint por validação e cache local em `data/processed/`.

## Relação com o problema de pesquisa

O pipeline permite comparar diferentes abordagens de classificação, avaliar o efeito do balanceamento e gerar explicações globais e locais dos modelos. Dessa forma, a implementação sustenta a análise experimental proposta para o TCC.

## Modelo deep learning pesado

O pipeline passou a incluir um experimento PyTorch mais robusto para records500, com selecao por validacao e avaliacao final unica no teste. Em ambientes com GPU, a configuracao permite executar mais de uma arquitetura e avaliar ensemble por media de probabilidades; em CPU, a execucao e limitada automaticamente para manter custo computacional controlado. O treinamento usa AdamW, scheduler por validacao, early stopping, checkpoint do melhor modelo, normalizacao calculada apenas no treino e estrategias de desbalanceamento aplicadas somente no treino.
