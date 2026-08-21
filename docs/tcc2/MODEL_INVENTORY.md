# Inventário de modelos do TCC II

## Modelos executáveis no ambiente principal

| Grupo | Modelos | Entrada | Frequência | Situação |
|---|---|---|---:|---|
| Clássicos | Regressão Logística, SVM, Random Forest, LightGBM e CatBoost | Atributos estatísticos e espectrais | 500 Hz | Pipelines multilabel preparados |
| Convolução temporal | TCN residual dilatada | Sinal bruto | 500 Hz | Treino, checkpoint e smoke test preparados |

## Implementações externas preservadas

| Fonte | Modelos/papel | Commit pinado | Licença | Ambiente |
|---|---|---|---|---|
| `helme/ecg_ptbxl_benchmarking` | inception1d, xresnet1d101, resnet1d_wang, fcn_wang, LSTM e BiLSTM | `cdbf4e66d7e57d9b6a2657b6024716212b8d0afa` | GPL-3.0 | Conda legado do próprio repositório |
| `tmehari/ssm_ecg` | S4 supervisionado e transferência CPC | `94b6cc708d70e4b832e91c8494f9595a320f8dd7` | MIT | Conda do próprio repositório |
| `kweimann/ECG-JEPA` | Pré-treinamento e fine-tuning JEPA | `504b4e5b55c2de3c6dca0492435b993a8f4a4013` | MIT | Ambiente PyTorch separado |
| `tmehari/ecg-selfsupervised` | CPC como contingência | `956c6c17c9496b2ba459638c613c6efc148b95df` | GPL-3.0 | Ambiente legado separado |

Os checkouts são criados por `scripts/fetch_external_repositories.py`. Nenhum código externo é copiado ou reimplementado silenciosamente em `src/tcc_ecg`. Isso permite distinguir reprodução original de extensão local.

Para Helme, `scripts/run_helme_validation.py` reutiliza diretamente as classes e configurações do checkout pinado. A única adaptação é a orquestração: o script oficial prediz o fold de teste para cada modelo, enquanto o adaptador do TCC II limita a seleção aos folds de treino e validação.

## Referência histórica

CNN simples, ResNet1D leve, InceptionTime adaptada e ResNet1D-SE permanecem como resultados do TCC I. Elas não substituem as implementações originais de Helme e não serão usadas para afirmar reprodução do benchmark.

## Extensões condicionais

ProtoECGNet, Wavelet+NN fora da reprodução principal, modelos fundacionais maiores, ensembles, segunda base externa e comparação com médicos dependem da conclusão do escopo obrigatório e de nova aprovação do orientador.
