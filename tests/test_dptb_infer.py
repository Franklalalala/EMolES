import torch
from emoles.dptb import process_dataset, setup_dataset, setup_db_path, setup_output_directory, load_model


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