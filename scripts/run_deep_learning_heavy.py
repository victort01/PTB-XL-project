"""Executa o experimento deep learning heavy sem imprimir logs longos."""

from tcc_ecg.config import load_config
from tcc_ecg.data import prepare_metadata
from tcc_ecg.dl_training import train_deep_learning_heavy


def main() -> None:
    config = load_config()
    metadata = prepare_metadata(config, save_summary=True)
    result = train_deep_learning_heavy(metadata, config)
    print(result["metrics"])


if __name__ == "__main__":
    main()
