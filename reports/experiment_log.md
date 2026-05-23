# Experiment Log


## Deep Learning Strong - InceptionTime1D

- Model: inceptiontime1d_strong
- Frequency: 500 Hz
- Loss: focal with class alpha
- Accuracy: 0.7824
- F1 macro: 0.6840
- Recall macro: 0.6806
- Precision macro: 0.7057
- Device: cpu
- Epochs trained: 30
- Best epoch: 12
- Training seconds: 27118.13
- Notes: Test fold evaluated only after checkpoint selection by validation F1 macro.

## Hardware Detectado - Experimento Heavy

- Timestamp: 2026-05-20 01:35:13
- Python .venv: 3.11.9
- PyTorch .venv: 2.12.0+cpu
- CUDA disponivel: False
- Dispositivo: CPU
- nvidia-smi: indisponivel no PATH
- Proximo experimento: deep_learning_heavy em records500, com limite automatico de uma execucao em CPU.

## Tentativa Heavy InceptionTime CPU interrompida

- Motivo: mais de 20 minutos sem concluir a primeira epoca em CPU.
- Acao: manter arquitetura implementada e executar alternativa ResNet1D-SE pesada, mais adequada ao hardware sem CUDA.


## Tentativa Heavy ResNet1D-SE 512 interrompida

- Motivo: primeira epoca ainda sem progresso em CPU apos mais de 10 minutos.
- Acao: reduzir largura maxima para 256 canais, mantendo arquitetura ResNet1D-SE, records500 e estrategias de desbalanceamento apenas no treino.


## Tentativa Heavy ResNet1D-SE 256 interrompida

- Motivo: primeira epoca sem progresso apos nova janela longa em CPU.
- Acao: reduzir largura para 48/96/192 canais com SE, preservando records500 e estrategias de desbalanceamento apenas no treino.


## Tentativa Heavy ResNet1D-SE 192 interrompida

- Motivo: custo ainda alto em CPU antes da primeira epoca registrada.
- Acao: usar ResNet1D-SE media com 32/64/128 canais e mais blocos residuais-SE, mantendo records500 e criterios metodologicos.


## Deep Learning Heavy

- Model: deep_learning_heavy_resnet1d_se
- Frequency: 500 Hz
- Device: cpu
- Architecture: resnet1d_se
- Loss: focal
- Weighted sampler: True
- Accuracy: 0.7248
- F1 macro: 0.6390
- Recall macro: 0.6677
- Best epoch: 71
- Training seconds: 49759.88
- Notes: test fold evaluated only after validation-based checkpoint/ensemble selection.
