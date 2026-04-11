import os
import shutil
from pathlib import Path


def setup_output_directory(output_path):
    """Prepare a clean output directory."""
    output_dir = Path(output_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()


def setup_db_path(db_path):
    """Return a clean absolute ASE DB path."""
    absolute_db_path = os.path.abspath(db_path)
    if os.path.exists(absolute_db_path):
        os.remove(absolute_db_path)
    return absolute_db_path
