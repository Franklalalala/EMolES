import os
import time
from pathlib import Path

import numpy as np
import torch
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.nn.hr2hk import HR2HK
from dptb.utils.argcheck import collect_cutoffs
from dptb.utils.tools import j_loader
from ase.db.core import connect
from ase.atoms import Atoms
from emoles.utils import setup_output_directory, setup_db_path


def process_batch_to_npy(batch_info, model, device, has_overlap, filename_prefix):
    """
    Process a batch through the model and save Hamiltonian and overlap matrices as NPY files.

    Args:
        batch_info: Dictionary containing batch data
        model: The DPTB model
        device: Computing device (CPU/CUDA)
        has_overlap: Boolean indicating if overlap matrices should be calculated
        filename_prefix: Prefix for output filenames
    """
    # Set k-point to gamma point (0,0,0)
    batch_info['kpoint'] = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Calculate Hamiltonian matrix
    ham_hr2hk = HR2HK(
        idp=model.idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device=device
    )
    ham_out_data = ham_hr2hk.forward(batch_info)
    hamiltonian = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
    ham_ndarray = hamiltonian.real.cpu().numpy()
    np.save(f'{filename_prefix}_ham.npy', ham_ndarray[0])

    # Calculate overlap matrix if required
    if has_overlap:
        overlap_hr2hk = HR2HK(
            idp=model.idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            overlap=True,
            device=device
        )
        overlap_out_data = overlap_hr2hk.forward(batch_info)
        overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
        overlap_ndarray = overlap.real.cpu().numpy()
        np.save(f'{filename_prefix}_overlap.npy', overlap_ndarray[0])


def save_batch_data(output_dir, idx, original_data, predicted_data, model, device, has_overlap):
    """
    Save original and predicted data to NPY files in a directory structure.

    Args:
        output_dir: Base output directory
        idx: Index/ID for the current batch
        original_data: Original input data
        predicted_data: Model predictions
        model: The DPTB model
        device: Computing device (CPU/CUDA)
        has_overlap: Boolean indicating if overlap matrices should be calculated
    """
    cwd = os.getcwd()
    batch_dir = Path(output_dir) / str(idx)
    batch_dir.mkdir(exist_ok=True)
    os.chdir(batch_dir)

    # Save original and predicted data
    process_batch_to_npy(
        batch_info=original_data,
        model=model,
        device=device,
        has_overlap=has_overlap,
        filename_prefix='original'
    )
    process_batch_to_npy(
        batch_info=predicted_data,
        model=model,
        device=device,
        has_overlap=has_overlap,
        filename_prefix='predicted'
    )

    os.chdir(cwd)


def save_atomic_structure(atomic_data, type_mapper, db_path):
    """
    Save atomic structure to an ASE database.

    Args:
        atomic_data: Dictionary containing atomic data
        type_mapper: Mapper to convert between indices and atomic numbers
        db_path: Path to the ASE database file
    """
    atomic_nums = atomic_data['atom_types'].cpu().reshape(-1)
    atomic_nums = type_mapper.untransform(atomic_nums).numpy()
    positions = atomic_data['pos'].cpu().numpy()
    atoms = Atoms(symbols=atomic_nums, positions=positions)

    with connect(db_path) as db:
        db.write(atoms)


def load_model(checkpoint_path, device):
    """
    Load the DPTB model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Computing device (CPU/CUDA)

    Returns:
        Loaded model on the specified device
    """
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    return model


def setup_dataset(config_path, reference_info, device):
    """
    Set up the dataset and data loader based on configuration.

    Args:
        config_path: Path to the configuration JSON file
        reference_info: Dictionary with reference dataset information
        device: Computing device (CPU/CUDA)

    Returns:
        data_loader: DataLoader for the dataset
        dataset: The dataset object
        cutoff_options: Dictionary of cutoff options
    """
    jdata = j_loader(config_path)
    cutoff_options = collect_cutoffs(jdata)

    dataset = build_dataset(
        **cutoff_options,
        **reference_info,
        **jdata["common_options"]
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    return data_loader, dataset, cutoff_options




def process_dataset(data_loader, dataset, model, device, output_dir, db_path, max_items=None):
    """
    Process the entire dataset through the model.

    Args:
        data_loader: DataLoader for the dataset
        dataset: The dataset object
        model: The DPTB model
        device: Computing device (CPU/CUDA)
        output_dir: Directory for output files
        db_path: Path to the ASE database file
        max_items: Maximum number of items to process (None for all)

    Returns:
        processing_time: Average processing time per item
    """
    start_time = time.time()
    type_mapper = dataset.type_mapper
    has_overlap = dataset.get_overlap

    for idx, batch in enumerate(data_loader):
        if max_items is not None and idx >= max_items:
            break

        # Convert batch to required format and move to device
        batch_dict = AtomicData.to_AtomicDataDict(batch.to(device))
        original_data = batch_dict.copy()

        # Get model predictions
        with torch.no_grad():
            predicted_data = model(batch_dict)

        # Save data to files
        save_batch_data(
            output_dir=output_dir,
            idx=idx,
            original_data=original_data,
            predicted_data=predicted_data,
            model=model,
            device=device,
            has_overlap=has_overlap
        )

        # Save atomic structure to database
        save_atomic_structure(
            atomic_data=original_data,
            type_mapper=type_mapper,
            db_path=db_path
        )

    end_time = time.time()
    processing_time = (end_time - start_time) / (idx + 1)
    return processing_time


if __name__ == "__main__":
    """Main function to execute the entire workflow."""
    # Configuration paths
    checkpoint_path = r'nnenv.best.pth'
    config_path = r'train_config.json'
    output_path = 'output'
    db_path = r'dump.db'

    # Reference dataset information
    reference_info = {
        "root": r"/share/lmk_1399/1104_no_li_workbase/1104_splited_dptb_format/6311gdp/test",
        "prefix": "data",
        "type": "LMDBDataset",
        "get_Hamiltonian": True,
        "get_overlap": False
    }

    # Set device (CPU or CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup output directory and database
    setup_output_directory(output_path)
    abs_db_path = setup_db_path(db_path)

    # Load model
    model = load_model(checkpoint_path, device)

    # Setup dataset and loader
    data_loader, dataset, cutoff_options = setup_dataset(config_path, reference_info, device)

    # Print cutoff options for debugging
    print('=' * 50)
    print('Cutoff options:')
    print(cutoff_options)
    print('=' * 50)

    # Process the dataset
    time_per_item = process_dataset(
        data_loader=data_loader,
        dataset=dataset,
        model=model,
        device=device,
        output_dir=output_path,
        db_path=abs_db_path,
        max_items=None  # Set a number to limit processing, None for all items
    )

    print(f'Average processing time per item: {time_per_item:.4f} seconds')