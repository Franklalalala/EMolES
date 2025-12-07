import os
import time
from pathlib import Path
import copy

import numpy as np
import torch
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only
from dptb.utils.argcheck import collect_cutoffs
from dptb.utils.tools import j_loader
from ase.db.core import connect
from ase.atoms import Atoms
from emoles.utils import setup_output_directory, setup_db_path
from dptb.nnops.loss import HamilLossAbsMAE
from tqdm import tqdm
from dptb.data.interfaces.ham_to_feature import feature_to_block
from dptb.postprocess import write_blocks_to_abacus_csr


# ==========================================
# Helper Functions
# ==========================================

def get_abacus_csr_name(matrix_symbol: str) -> str:
    """根据 matrix_symbol 返回标准的 Abacus CSR 文件名。"""
    symbol_map = {
        'DM': 'dmrs1_nao.csr',
        'H': 'hrs1_nao.csr',
        'S': 'srs1_nao.csr'
    }
    return symbol_map.get(matrix_symbol, f'{matrix_symbol.lower()}rs1_nao.csr')


def resolve_save_path(base_dir: Path, prefix: str, filename: str, split_dirs: bool) -> str:
    """解析保存路径。"""
    if split_dirs:
        target_dir = base_dir / prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir / filename)
    else:
        return str(base_dir / f"{prefix}_{filename}")


def convert_to_numpy_recursive(obj):
    """递归将 Tensor 转换为 Numpy array，用于存入 ASE DB。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: convert_to_numpy_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_numpy_recursive(v) for v in obj]
    return obj


# ==========================================
# Core Processing Logic
# ==========================================

def process_batch_data(batch_info, model, device, output_dir: Path, prefix: str,
                       has_overlap: bool, split_dirs: bool,
                       save_npy: bool = False, save_csr_info: dict = None,
                       provided_blocks: dict = None):
    """
    Process a batch: Save matrices to Disk (CSR/NPY).
    """
    # 1. Save CSR to Disk
    if save_csr_info:
        matrix_symbol = save_csr_info.get('matrix_symbol', 'H')
        csr_filename = get_abacus_csr_name(matrix_symbol)
        output_path_csr = resolve_save_path(output_dir, prefix, csr_filename, split_dirs)

        # 核心优化：复用传入的 block，避免重复计算
        if provided_blocks is not None:
            block = provided_blocks
        else:
            block = feature_to_block(data=batch_info, idp=model.idp)

        write_blocks_to_abacus_csr(
            matrix_symbol=matrix_symbol,
            atomic_numbers=save_csr_info['atomic_numbers'],
            basis_dict=save_csr_info['basis_dict'],
            blocks_dict=block,
            output_path=output_path_csr,
        )

    # 2. Save NPY to Disk
    if save_npy:
        # Set k-point to gamma point
        batch_info['kpoint'] = torch.tensor([0.0, 0.0, 0.0], device=device)

        # Hamiltonian (H)
        ham_hr2hk = HR2HK_Gamma_Only(
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

        npy_filename = 'ham.npy' if not save_csr_info else f"{save_csr_info.get('matrix_symbol', 'ham')}.npy"
        output_path_npy = resolve_save_path(output_dir, prefix, npy_filename, split_dirs)
        np.save(output_path_npy, ham_ndarray)

    # Overlap (S) - Only if needed
    if has_overlap and save_npy:
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

        output_path_s = resolve_save_path(output_dir, prefix, 'overlap.npy', split_dirs)
        np.save(output_path_s, overlap_ndarray[0])


def save_batch_wrapper(output_dir, idx, original_data, predicted_data, model, device, has_overlap,
                       save_csr_info: dict = None, save_npy: bool = True,
                       save_disk_original: bool = True, split_dirs: bool = False,
                       original_blocks: dict = None, predicted_blocks: dict = None):
    """
    负责将数据保存到磁盘（CSR/NPY 文件）。
    """
    cwd = os.getcwd()
    batch_dir = Path(output_dir) / str(idx)
    batch_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(batch_dir)

    current_save_dir = Path('.')

    # Save Original to Disk
    if save_disk_original:
        process_batch_data(
            batch_info=original_data,
            model=model,
            device=device,
            output_dir=current_save_dir,
            prefix='original',
            has_overlap=has_overlap,
            split_dirs=split_dirs,
            save_npy=save_npy,
            save_csr_info=save_csr_info,
            provided_blocks=original_blocks
        )

    # Save Predicted to Disk
    process_batch_data(
        batch_info=predicted_data,
        model=model,
        device=device,
        output_dir=current_save_dir,
        prefix='predicted',
        has_overlap=has_overlap,
        split_dirs=split_dirs,
        save_npy=save_npy,
        save_csr_info=save_csr_info,
        provided_blocks=predicted_blocks
    )

    os.chdir(cwd)


def save_atomic_structure(atomic_data, atomic_nums, db_path, an_err=None, additional_data=None):
    """将原子结构和额外数据存入 ASE DB。"""
    positions = atomic_data['pos'].cpu().numpy()

    atoms = Atoms(symbols=atomic_nums, positions=positions)
    pbc = atomic_data['pbc'].cpu().numpy()[0]
    if pbc.any():
        atoms.pbc = pbc
        cell = atomic_data['cell'].cpu().numpy()
        atoms.cell = cell[0]

    # 构建 data 字典
    data_content = {}
    if an_err is not None:
        data_content['err'] = an_err

    if additional_data:
        data_content.update(additional_data)

    with connect(db_path) as db:
        if data_content:
            db.write(atoms, data=data_content)
        else:
            db.write(atoms)
    return atoms


def load_model(checkpoint_path, device):
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    return model


def setup_dataset(config_path, reference_info, device, batch_size=1):
    jdata = j_loader(config_path)
    cutoff_options = collect_cutoffs(jdata)
    dataset = build_dataset(**cutoff_options, **reference_info, **jdata["common_options"])
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return data_loader, dataset, cutoff_options


def extract_csr_from_db(db_path: str, output_root: str,
                        extract_predicted: bool = True,
                        extract_original: bool = False,
                        split_dirs: bool = True):
    """
    遍历 ase.db，提取 blocks 和 csr_info，还原为 .csr 文件。

    :param db_path: ase.db 文件路径
    :param output_root: 输出根目录 (e.g. 'restored_output')
    :param extract_predicted: 是否提取预测数据
    :param extract_original: 是否提取原始数据 (如果 db 中有的话)
    :param split_dirs: 是否使用分层目录结构 (output/0/predicted/...)
    """
    db_path = Path(db_path)
    output_root = Path(output_root)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # 连接数据库
    with connect(db_path) as db:
        total_rows = db.count()
        print(f"Found {total_rows} entries in {db_path}")

        # 遍历数据库
        # 注意：ASE DB row.id 通常从 1 开始，这里使用 enumerate 保持 0-based 索引以匹配之前的 output/0 逻辑
        for idx, row in enumerate(tqdm(db.select(), total=total_rows, desc="Extracting CSR")):

            data = row.data

            # 检查是否有必要的信息
            if 'csr_info' not in data:
                print(f"Skipping row {idx}: No 'csr_info' found in data.")
                continue

            csr_info = data['csr_info']
            matrix_symbol = csr_info.get('matrix_symbol', 'H')
            basis_dict = csr_info.get('basis_dict')

            # 从 row 中获取原子序数 (比从 data['csr_info'] 获取更稳健，因为它是 ASE 原生字段)
            atomic_numbers = row.numbers

            # 准备当前条目的输出目录
            batch_dir = output_root / str(idx)
            batch_dir.mkdir(parents=True, exist_ok=True)

            csr_filename = get_abacus_csr_name(matrix_symbol)

            # --- 1. 提取并保存 Predicted Data ---
            if extract_predicted and 'predicted_data' in data:
                blocks = data['predicted_data']
                save_path = resolve_save_path(batch_dir, 'predicted', csr_filename, split_dirs)

                write_blocks_to_abacus_csr(
                    matrix_symbol=matrix_symbol,
                    atomic_numbers=atomic_numbers,
                    basis_dict=basis_dict,
                    blocks_dict=blocks,
                    output_path=save_path
                )

            # --- 2. 提取并保存 Original Data ---
            if extract_original and 'original_data' in data:
                blocks = data['original_data']
                save_path = resolve_save_path(batch_dir, 'original', csr_filename, split_dirs)

                write_blocks_to_abacus_csr(
                    matrix_symbol=matrix_symbol,
                    atomic_numbers=atomic_numbers,
                    basis_dict=basis_dict,
                    blocks_dict=blocks,
                    output_path=save_path
                )


def process_dataset(data_loader, dataset, model, device, output_dir, db_path, max_items=None,
                    err_info_file: str = 'err.txt', save_data_flag: bool = True,
                    save_csr_info: dict = None, save_npy_flag: bool = False,
                    save_disk_files: bool = True, save_disk_original: bool = True,
                    split_dirs: bool = False,
                    save_db_original: bool = False, save_db_predicted: bool = True):
    """
    处理数据集，包含保存到 DB 和保存到磁盘（output_dir）的逻辑。
    """
    start_time = time.time()
    type_mapper = dataset.type_mapper
    has_overlap = dataset.get_overlap
    loss_list = []

    # 逻辑判断：是否真的进行磁盘写入
    # 只有当 output_dir 存在 且 save_disk_files 为 True 且 (有保存NPY或CSR的需求) 时，才为 True
    should_save_to_disk = (output_dir is not None) and save_disk_files and (save_npy_flag or save_csr_info is not None)

    total_batches = len(data_loader) if hasattr(data_loader, '__len__') else None
    if max_items is not None and total_batches is not None:
        total_batches = min(total_batches, int(max_items / data_loader.batch_size))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, desc="Processing", unit="batch", total=total_batches)):
            if total_batches is not None and idx == total_batches:
                break

            batch_dict = AtomicData.to_AtomicDataDict(batch.to(device))
            original_data = copy.deepcopy(batch_dict)

            loss_func = HamilLossAbsMAE(idp=model.hamiltonian.idp)
            predicted_data = model(batch_dict)

            a_loss = loss_func(predicted_data, original_data)
            a_loss_val = a_loss.cpu().numpy()
            loss_list.append(a_loss_val)

            if save_data_flag:
                assert data_loader.batch_size == 1, 'Currently, the save batch only support batch size = 1'

                atomic_nums = original_data['atom_types'].cpu().reshape(-1)
                atomic_nums = type_mapper.untransform(atomic_nums).numpy()

                # 更新 CSR 保存所需的原子序数信息
                if save_csr_info:
                    save_csr_info.update({'atomic_numbers': atomic_nums})

                # -----------------------------------------------
                # 1. 智能计算 Blocks (避免重复，按需计算)
                # -----------------------------------------------
                # 判断条件：(存DB需要) 或者 (存磁盘需要 CSR)

                # A. 预测数据 Blocks
                need_pred_blocks = save_db_predicted or (should_save_to_disk and save_csr_info)
                pred_blocks_tensor = None
                if need_pred_blocks:
                    pred_blocks_tensor = feature_to_block(data=predicted_data, idp=model.idp)

                # B. 原始数据 Blocks
                need_orig_blocks = save_db_original or (should_save_to_disk and save_disk_original and save_csr_info)
                orig_blocks_tensor = None
                if need_orig_blocks:
                    orig_blocks_tensor = feature_to_block(data=original_data, idp=model.idp)

                # -----------------------------------------------
                # 2. 存入 ASE DB (将 Tensor Block 转为 Numpy)
                # -----------------------------------------------
                db_extra_data = {}

                if save_csr_info:
                    db_extra_data['csr_info'] = copy.deepcopy(save_csr_info)

                if save_db_predicted and pred_blocks_tensor is not None:
                    db_extra_data['predicted_data'] = convert_to_numpy_recursive(pred_blocks_tensor)

                if save_db_original and orig_blocks_tensor is not None:
                    db_extra_data['original_data'] = convert_to_numpy_recursive(orig_blocks_tensor)

                save_atomic_structure(
                    atomic_data=original_data,
                    atomic_nums=atomic_nums,
                    db_path=db_path,
                    an_err=a_loss_val,
                    additional_data=db_extra_data
                )

                # -----------------------------------------------
                # 3. 存入 磁盘文件 (仅当 output_dir 存在且 switch 打开)
                # -----------------------------------------------
                if should_save_to_disk:
                    save_batch_wrapper(
                        output_dir=output_dir,
                        idx=idx,
                        original_data=original_data,
                        predicted_data=predicted_data,
                        model=model,
                        device=device,
                        has_overlap=has_overlap,
                        save_csr_info=save_csr_info,
                        save_npy=save_npy_flag,
                        save_disk_original=save_disk_original,
                        split_dirs=split_dirs,
                        original_blocks=orig_blocks_tensor,  # 复用
                        predicted_blocks=pred_blocks_tensor  # 复用
                    )

            torch.cuda.empty_cache()

        end_time = time.time()
        processing_time = (end_time - start_time) / (idx + 1)
        loss_arr = np.array(loss_list)
        return processing_time, float(loss_arr.mean())


def process_dataset_dip(data_loader, dataset, model, device, output_dir, db_path, loss_func, max_items=None):
    # 此部分保持简略，如需扩展可参考 process_dataset
    start_time = time.time()
    loss_list = []
    # ... (省略具体实现，保持与上方结构一致)
    return 0.0, 0.0


if __name__ == "__main__":
    # =========================================================
    # Path Configuration
    # =========================================================
    checkpoint_path = r'nnenv.best.pth'
    config_path = r'train_config.json'

    # OUTPUT DIR: 设为 None 则完全关闭磁盘文件写入 (CSR/NPY)
    # output_path = None
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

    # =========================================================
    # Control Flags (配置区)
    # =========================================================

    # 1. 磁盘存储总开关 (CSR / NPY)
    # 如果 output_path 为 None，此开关自动视为 False
    save_disk_files = True

    # 2. 具体磁盘格式控制
    save_npy_flag = False  # 是否保存 NPY
    save_disk_original = False  # 磁盘上是否保存原始数据 (CSR/NPY)
    split_dirs = True  # 是否分层级存储 (output/0/predicted/...)

    # 3. CSR 配置 (用于磁盘存储 和 DB存储)
    # 如果不需要处理CSR相关逻辑，设为 None
    # save_csr_config = None
    # 注意：save_csr_config 必须定义，因为 dataset 初始化需要它，即使不存磁盘也要存DB
    save_csr_config_template = {'matrix_symbol': 'H'}

    # 4. ASE DB 存储控制 (存入 .db 文件)
    save_db_predicted = True  # 是否将预测 Block (Numpy) 存入 DB
    save_db_original = False  # 是否将原始 Block (Numpy) 存入 DB

    # =========================================================
    # Setup & Run
    # =========================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if output_path is not None and save_disk_files:
        setup_output_directory(output_path)

    abs_db_path = setup_db_path(db_path)

    model = load_model(checkpoint_path, device)
    data_loader, dataset, cutoff_options = setup_dataset(config_path, reference_info, device)

    # 补全 CSR config 的 basis
    if save_csr_config_template:
        save_csr_config_template['basis_dict'] = dataset.basis_dict

    print('=' * 50)
    print(f'Cutoff options: {cutoff_options}')
    print(f'Output Dir: {output_path}')
    print(f'Save Disk Files: {save_disk_files}')
    print('=' * 50)

    time_per_item = process_dataset(
        data_loader=data_loader,
        dataset=dataset,
        model=model,
        device=device,
        output_dir=output_path,
        db_path=abs_db_path,
        max_items=None,
        save_csr_info=save_csr_config_template,
        save_npy_flag=save_npy_flag,
        save_disk_files=save_disk_files,  # 传入总开关
        save_disk_original=save_disk_original,
        split_dirs=split_dirs,
        save_db_original=save_db_original,
        save_db_predicted=save_db_predicted
    )

    print(f'Average processing time per item: {time_per_item:.4f} seconds')