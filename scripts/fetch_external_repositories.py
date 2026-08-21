"""Busca implementacoes externas nos commits registrados pelo projeto."""

from __future__ import annotations

import argparse

from tcc_ecg.external_repositories import (
    fetch_repository,
    load_repository_specs,
    repository_inventory,
    write_repository_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/external_repositories.yaml")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    specs = load_repository_specs(args.config)
    selected = args.only or list(specs)
    for name in selected:
        if name not in specs:
            raise KeyError(f"Repositorio desconhecido: {name}")
        destination = fetch_repository(name, specs[name])
        print(f"{name}: {destination}")
    output = write_repository_inventory(repository_inventory(specs))
    print(f"Inventario: {output}")


if __name__ == "__main__":
    main()

