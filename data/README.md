# Dados

Este projeto espera que o PTB-XL seja baixado manualmente e posicionado em:

```text
data/raw/ptb-xl/
```

Dependendo da forma de extração, o PhysioNet pode criar uma subpasta com o nome completo do dataset. Nesta execução local, `configs/config.yaml` aponta para:

```text
data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

A pasta configurada deve conter, no mínimo:

```text
ptbxl_database.csv
scp_statements.csv
records100/
```

Opcionalmente, para experimentos futuros em maior resolução:

```text
records500/
```

Em `configs/config.yaml`, `data.signal_frequency: 100` usa `filename_lr` e `records100/`; `data.signal_frequency: 500` usa `filename_hr` e `records500/`.

Os dados brutos, arquivos processados e modelos treinados não são versionados no Git.
