# Protocolo experimental do TCC II

## Tarefa principal

- Formulação: multilabel.
- Rótulos: `NORM`, `MI`, `STTC`, `CD` e `HYP`.
- Registros com mais de uma superclasse são mantidos.
- A formulação multiclasse `strict_single_label` do TCC I permanece apenas como referência histórica.

## Divisão dos dados

- Treino: folds 1 a 8.
- Validação: fold 9.
- Teste interno: fold 10.
- Normalização, imputação, pesos e aumentações são ajustados somente no treino.
- Thresholds por classe são escolhidos somente no fold 9.
- O teste exige checkpoint congelado, hash de configuração compatível e flag explícita.

## Métricas

A métrica principal é macro AUROC, para manter compatibilidade com o benchmark multilabel do PTB-XL. Também devem ser reportados macro AUPRC, F1 macro, F1 micro, precisão macro, recall macro, acurácia de subconjunto, Hamming loss e métricas por classe. A acurácia de subconjunto não deve ser comparada diretamente à acurácia multiclasse do TCC I.

## Frequências e comparabilidade

- Benchmark original de Helme: 100 Hz, conforme a implementação de referência.
- Modelos clássicos aprimorados: 500 Hz e atributos estatísticos/espectrais.
- TCN e extensões locais: 500 Hz quando suportado.
- S4 e ECG-JEPA: frequência definida pela implementação e checkpoint originais.

A comparação representa configurações experimentais completas. Ela não isola apenas o efeito do algoritmo, porque representação, frequência, pré-treinamento e estratégia de otimização podem diferir.

## Congelamento

Cada candidato produz um `candidate_manifest.json` com checkpoint, configuração, commit, métricas de validação e thresholds. O comando `freeze` transforma o candidato selecionado em um manifesto congelado. O comando `evaluate-test` verifica o hash do checkpoint e bloqueia uma segunda avaliação, salvo mudança deliberada e documentada do protocolo.

## Validação externa

A base externa permanece fora do treino, da seleção e da definição de thresholds. Georgia é a primeira candidata, condicionada à auditoria de rótulos, cobertura, domínio, licença e ausência de participação no pré-treinamento do checkpoint. A avaliação externa acontece somente depois do teste interno e não autoriza reajuste do modelo.

