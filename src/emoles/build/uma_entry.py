from emoles.build.uma_core import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_WORKSPACE,
    smiles_to_atoms,
)
from emoles.build.uma_serial import entry, main

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_WORKSPACE",
    "entry",
    "main",
    "smiles_to_atoms",
]


if __name__ == "__main__":
    main()
