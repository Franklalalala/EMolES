import multiprocessing as mp

from emoles.build.uma_core import DEFAULT_CHECKPOINT, DEFAULT_MODEL_NAME, DEFAULT_WORKSPACE, smiles_to_atoms
from emoles.build.uma_parallel import entry, main

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_WORKSPACE",
    "entry",
    "main",
    "smiles_to_atoms",
]


if __name__ == "__main__":
    mp.freeze_support()
    main()
