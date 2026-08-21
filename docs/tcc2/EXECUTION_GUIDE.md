# Guia de execução do TCC II

## 1. Clonar e instalar

```bash
git clone https://github.com/victort01/PTB-XL-project.git
cd PTB-XL-project
git switch tcc2-development
python -m venv .venv
```

Ative a `.venv` e execute:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,tcc2]"
python -m pytest -q
python scripts/train_tcc2.py smoke --config configs/tcc2_multilabel.yaml
```

Para CUDA, instale primeiro a distribuição de PyTorch indicada para o driver da máquina e depois instale o projeto. Confirme com:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Posicionar dados

Defina `PTBXL_DATA_DIR` no ambiente ou em um arquivo `.env` local. O diretório deve conter `ptbxl_database.csv`, `scp_statements.csv`, `records100/` e `records500/`.

As bases externas usam `GEORGIA_DATA_DIR`, `CODE15_DATA_DIR`, `MIMIC_IV_ECG_DATA_DIR` e `ICBEB2018_DATA_DIR`. Não é necessário possuir todas. O comando de auditoria identifica o que está disponível sem executar modelos.

## 3. Auditorias iniciais

```bash
python scripts/audit_external_dataset.py --config configs/tcc2_multilabel.yaml
python scripts/fetch_external_repositories.py --config configs/external_repositories.yaml
```

O segundo comando cria `external/` em commits destacados e gera um inventário. Essa pasta não é versionada.

## 4. Modelos clássicos

```bash
python scripts/prepare_tcc2_features.py --config configs/tcc2_multilabel.yaml
python scripts/train_tcc2.py train --config configs/tcc2_multilabel.yaml --model lightgbm
```

Substitua o modelo por `logistic_regression`, `svm`, `random_forest` ou `catboost`. Cada execução salva somente métricas de validação e um manifesto de candidato.

## 5. TCN

```bash
python scripts/train_tcc2.py train --config configs/tcc2_multilabel.yaml --model tcn
```

O cache memmap evita releitura de WFDB em cada época. A normalização é calculada somente nos índices dos folds 1 a 8. `last.pt` permite retomada e `best.pt` acompanha a melhor macro AUROC de validação.

## 6. Benchmark original de Helme

Depois de `fetch_external_repositories`, crie o ambiente fornecido pelo próprio projeto:

```bash
conda env create -f external/ecg_ptbxl_benchmarking/ecg_env.yml
conda activate ecg_env
```

O benchmark usa Python 3.8, FastAI 1.0.61 e PyTorch 1.4 no arquivo original. Execute a preparação e os experimentos a partir do checkout pinado. Não instale esse ambiente legado dentro da `.venv` principal.

O `reproduce_results.py` original produz predições de teste para todos os modelos. Para preservar o fold 10 durante a seleção, use o adaptador do projeto:

```bash
python scripts/run_helme_validation.py \
  --repository external/ecg_ptbxl_benchmarking \
  --data-dir /caminho/para/ptb-xl \
  --output-dir models/tcc2/helme \
  --model helme_inception1d
```

O adaptador importa a classe e a configuração originais, mas indexa apenas folds 1--8 e 9. Ele não chama o método `perform()` do benchmark, pois esse método também prediz o teste. As demais opções são `helme_xresnet1d101`, `helme_resnet1d_wang`, `helme_fcn_wang`, `helme_lstm` e `helme_lstm_bidir`.

## 7. S4 e ECG-JEPA

S4 deve usar `external/ssm_ecg/environment.yml`. ECG-JEPA deve usar seu `requirements.txt` em ambiente separado. Antes de usar checkpoints pré-treinados, registre os datasets de pré-treinamento e confirme que a base escolhida para validação externa não participou deles.

## 8. Congelar e liberar o teste

Compare todos os candidatos pelo fold 9. Depois de selecionar um:

```bash
python scripts/train_tcc2.py freeze --config configs/tcc2_multilabel.yaml \
  --candidate-manifest models/tcc2/tcn/seed_42/candidate_manifest.json
```

Somente então:

```bash
python scripts/train_tcc2.py evaluate-test --config configs/tcc2_multilabel.yaml \
  --frozen-manifest reports/manifests/tcc2/frozen_tcn.json
```

O manifesto registra hash do checkpoint, configuração, commit, thresholds derivados da validação e uso do teste. Não altere o protocolo após observar o fold 10.

## 9. Ordem recomendada

1. Testes e smoke tests.
2. Auditoria da Georgia e dos checkpoints pré-treinados.
3. Preparação de caches e atributos.
4. Reprodução de Helme em 100 Hz.
5. Modelos clássicos em 500 Hz.
6. TCN, S4 e ECG-JEPA.
7. Múltiplas sementes apenas nos finalistas.
8. Congelamento pelo fold 9.
9. Fold 10 uma única vez.
10. Validação externa do candidato congelado.
11. SHAP/LIME no tabular e Integrated Gradients no profundo.

## 10. Perfil qualitativo de recursos

- Modelos clássicos: prioridade para CPU e memória; paralelização por estimadores quando suportada.
- Helme em 100 Hz: GPU recomendada, mas com ambiente legado isolado.
- TCN em 500 Hz: GPU recomendada e cache memmap; batch ajustável em `training.batch_size`.
- S4: GPU recomendada e dependências específicas do checkout original.
- ECG-JEPA: GPU fortemente recomendada; fine-tuning deve partir de checkpoint cuja proveniência tenha sido auditada.

Se houver limitação de recursos, reduza primeiro buscas e sementes excedentes. Não altere os folds, o bloqueio do teste ou a validação externa para economizar processamento.
