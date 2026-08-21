# Classificação Multiclasse de ECGs PTB-XL

Projeto notebook-first para um TCC sobre classificação multiclasse de eletrocardiogramas de 12 derivações usando o dataset PTB-XL, modelos clássicos de Machine Learning, balanceamento de dados e interpretabilidade.

O estado consolidado do TCC I permanece na branch `tcc1-final`. O desenvolvimento do TCC II ocorre na branch `tcc2-development` e adiciona uma trilha multilabel separada, sem alterar os resultados históricos.

## Objetivo Acadêmico

Comparar Regressão Logística, SVM, Random Forest, LightGBM, CatBoost e um baseline simples de deep learning para classificar ECGs nas superclasses diagnósticas `NORM`, `MI`, `STTC`, `CD` e `HYP`. A métrica principal é F1-score macro, pois o problema é desbalanceado. A acurácia também é reportada.

## Estrutura

```text
configs/          configuracao reprodutivel
data/             dados brutos, intermediarios e processados nao versionados
notebooks/        fluxo principal do TCC, em ordem numerica
reports/          figuras, tabelas e textos para LaTeX
models/           modelos treinados nao versionados
src/tcc_ecg/      codigo reutilizavel
tests/            testes pequenos sem depender do PTB-XL completo
```

## TCC II: protocolo multilabel

O protocolo principal do TCC II está em `configs/tcc2_multilabel.yaml` e mantém as cinco superclasses `NORM`, `MI`, `STTC`, `CD` e `HYP` como rótulos binários simultâneos. Ele usa treino nos folds 1--8, validação no fold 9 e teste interno no fold 10. A seleção, os thresholds e o congelamento usam somente a validação; o fold 10 exige um manifesto congelado e um comando explícito.

Instalação recomendada em uma clonagem limpa:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,tcc2]"
# Alternativa equivalente: python -m pip install -r requirements-tcc2.txt
python -m pytest -q
python scripts/train_tcc2.py smoke --config configs/tcc2_multilabel.yaml
```

Fluxo operacional:

```bash
# Auditoria sem inferência das bases externas configuradas
python scripts/audit_external_dataset.py --config configs/tcc2_multilabel.yaml

# Checkouts independentes e pinados de Helme, S4, ECG-JEPA e CPC
python scripts/fetch_external_repositories.py --config configs/external_repositories.yaml

# Atributos multilabel dos modelos clássicos em 500 Hz
python scripts/prepare_tcc2_features.py --config configs/tcc2_multilabel.yaml

# Treino de um candidato usando somente treino e validação
python scripts/train_tcc2.py train --config configs/tcc2_multilabel.yaml --model tcn

# Congelamento após comparação pelo fold 9
python scripts/train_tcc2.py freeze --config configs/tcc2_multilabel.yaml \
  --candidate-manifest models/tcc2/tcn/seed_42/candidate_manifest.json

# Somente depois do congelamento formal
python scripts/train_tcc2.py evaluate-test --config configs/tcc2_multilabel.yaml \
  --frozen-manifest reports/manifests/tcc2/frozen_tcn.json
```

Os repositórios externos não são copiados para este Git. Eles são obtidos em `external/`, que é ignorado, e posicionados nos commits registrados em `configs/external_repositories.yaml`. As implementações originais de Helme e S4 possuem ambientes legados próprios e não são silenciosamente substituídas pelas arquiteturas reduzidas do TCC I.

Consulte `docs/tcc2/EXECUTION_GUIDE.md`, `docs/tcc2/PROTOCOL.md` e `docs/tcc2/MODEL_INVENTORY.md` antes dos experimentos longos.

## Configuração do Ambiente

Use uma `.venv` local para evitar instalar dependências no Python global.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Se o Python padrão do Windows for muito novo para o TensorFlow, crie a `.venv` com uma versão compatível instalada, por exemplo:

```bash
py -3.11 -m venv .venv
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Para executar deep learning, instale o extra `dl` dentro da `.venv`:

```bash
python -m pip install -e ".[dl]"
```

O extra `dl` instala as dependências de deep learning usadas no projeto, incluindo PyTorch para a ResNet1D e TensorFlow/Keras para o baseline simples. TensorFlow não faz parte da instalação mínima.

## Organização do PTB-XL

Baixe o PTB-XL manualmente e posicione os arquivos em:

```text
data/raw/ptb-xl/
```

Se o download for extraído diretamente do PhysioNet, pode haver uma subpasta com o nome da versão do dataset. Nesta execução local, a configuração aponta para:

```text
data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

Essa pasta deve conter:

```text
ptbxl_database.csv
scp_statements.csv
records100/
```

O projeto começa com `records100` por viabilidade computacional. A configuração permite mudar para `records500` em `configs/config.yaml`. Dados, caches processados e modelos grandes não são versionados.

## Frequência dos Sinais

A frequência dos sinais é controlada por `data.signal_frequency` em `configs/config.yaml`:

```yaml
data:
  signal_frequency: 100
```

- `100`: usa a coluna `filename_lr` e os arquivos em `records100/`.
- `500`: usa a coluna `filename_hr` e os arquivos em `records500/`.

`records100` é recomendado para desenvolvimento, depuração e validação rápida do pipeline. `records500` é recomendado para experimento final ou comparação complementar, pois tem maior resolução temporal, mas exige mais tempo de leitura, mais RAM e maior custo de processamento.

Após executar o notebook de treinamento para as duas frequências, gere a comparação com:

```bash
python scripts/compare_frequencies.py
```

O script salva:

```text
reports/tables/frequency_comparison.csv
reports/tables/frequency_comparison.tex
```

## Ordem dos Notebooks

1. `00_environment_check.ipynb`
2. `01_data_loading_and_labels.ipynb`
3. `02_eda.ipynb`
4. `03_feature_extraction.ipynb`
5. `04_train_classical_models.ipynb`
6. `05_deep_learning_baseline.ipynb`
7. `05b_deep_learning_resnet1d.ipynb`
8. `05c_deep_learning_strong.ipynb`
9. `06_evaluation_and_comparison.ipynb`
10. `07_interpretability.ipynb`

Os notebooks importam funções de `src/tcc_ecg` e salvam tabelas/figuras em `reports/`.

## Rótulos Multiclasse

O PTB-XL é originalmente multi-label. A estratégia padrão (`strict_single_label`) mantém apenas registros com exatamente uma superclass diagnóstica entre `NORM`, `MI`, `STTC`, `CD` e `HYP`. A estratégia alternativa (`primary_by_scp_weight`) escolhe a superclass com maior peso em `scp_codes` e deve ser usada apenas para análise de sensibilidade.

## Tratamento de Idade

No PTB-XL, `age == 300` representa anonimização de pacientes com idade real maior ou igual a 90 anos. O projeto cria:

- `age_is_anon_90_plus`: flag booleana;
- `age_clean`: idade com `300` substituído por `NaN`.

A imputação de `age_clean` ocorre somente dentro dos pipelines ajustados no treino, evitando interpretar 300 como idade fisiológica real.

## Folds e Vazamento de Dados

O projeto usa `strat_fold` do PTB-XL:

- treino: folds 1 a 8;
- validação: fold 9;
- teste: fold 10.

SMOTE, imputação e normalização são ajustados apenas no treino. O teste é usado somente na avaliação final.

## Métricas

São reportadas:

- accuracy;
- balanced accuracy;
- precision macro;
- recall macro;
- F1 macro;
- F1 weighted;
- classification report;
- matriz de confusão.

F1 macro é a métrica principal por tratar todas as classes com o mesmo peso.

## Deep Learning Residual

Além do baseline simples, o projeto inclui `notebooks/05b_deep_learning_resnet1d.ipynb`, que treina uma ResNet1D leve em PyTorch sobre sinais brutos. Por padrão, essa etapa usa `records500`, pois a maior resolução temporal pode preservar padrões do ECG que se perdem em features estatísticas agregadas.

A ResNet1D usa:

- normalização por canal calculada somente no treino;
- folds oficiais do PTB-XL;
- `class_weight` para desbalanceamento;
- `EarlyStopping` por `val_loss`;
- `ReduceLROnPlateau`;
- checkpoint do melhor modelo em `models/resnet1d_best.pt`.

Essa arquitetura é uma extensão controlada para comparação acadêmica. Ela não deve ser interpretada como evidência clínica definitiva.

O notebook `notebooks/05c_deep_learning_strong.ipynb` adiciona uma arquitetura InceptionTime 1D mais forte, ainda baseada em PyTorch e `records500`. Ela usa kernels temporais múltiplos, conexões residuais, aumentações leves apenas no treino, focal loss/class weights configuráveis, checkpoint por validação e avaliação única no teste.

## Comandos

```bash
make test
make lint
make clean-artifacts
```

## Limitações Conhecidas

- Features estatísticas simples são um baseline reproduzível, mas não capturam toda a morfologia clínica do ECG.
- A tarefa multiclasse descarta registros multi-label na estratégia principal.
- O baseline CNN é simples por desenho e não substitui uma arquitetura clínica especializada.
- Interpretabilidade com SHAP/LIME indica importância para o modelo, não causalidade médica.

## Próximos Passos

- Executar notebooks com o PTB-XL completo.
- Comparar `records100` e `records500`.
- Avaliar a estratégia alternativa de rótulo.
- Refinar features ou CNN se a meta de 80% de acurácia não for atingida.
- Considerar formulação multi-label em trabalho futuro.
