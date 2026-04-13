from emoles.inference.infer_entry import dptb_infer_from_ase_db


if __name__ == "__main__":
    db_path = r'/home/mingkang_nt/dptb_loss_workspace/cluster/16_slow_slow/test_1213/dump.db'
    checkpoint_path = r'/home/mingkang_nt/dptb_loss_workspace/cluster/16_slow_slow/checkpoint/nnenv.iter248001.pth'
    output_path = 'output'

    dptb_infer_from_ase_db(
        ase_db_path=db_path,
        out_path=output_path,
        checkpoint_path=checkpoint_path,
        limit=5,
        device='cuda'
    )
