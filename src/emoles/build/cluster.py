from typing import List, Tuple, Dict, Union
import numpy as np
import random
import os

from ase import Atoms
from ase.io import read, write

from rdkit import Chem
from scipy.spatial import ConvexHull, QhullError

from emoles.build.CombineMols3D import (
    get_bond_length,
    calculate_intermolecular_repulsion,
    DEFAULT_CLASH_FACTOR,
)
from emoles.build.patch_picker import (
    get_patch_atoms_and_indices,
    atom_2_mol,
)

# =============================================================================
# Configuration
# =============================================================================

# High penalty added to scoring when clashes are detected (does not change geometry)
CLASH_PENALTY_FOR_SCORING = 1e10


# =============================================================================
# Utilities
# =============================================================================

def is_xyz_path(text: str) -> bool:
    """Return True if the string looks like a .xyz file path."""
    return isinstance(text, str) and text.lower().endswith(".xyz")


def make_cache_key(identifier: Union[str, Atoms, Chem.Mol]) -> str:
    """Create a stable cache key for SMILES, XYZ path, ASE Atoms, or RDKit Mol inputs."""
    if isinstance(identifier, str):
        return f"XYZ:{identifier}" if is_xyz_path(identifier) else f"SMILES:{identifier}"
    if isinstance(identifier, Atoms):
        # Use object's memory address for a unique key per instance.
        return f"ASE@{id(identifier)}"
    if isinstance(identifier, Chem.Mol):
        return f"RDKIT@{id(identifier)}"
    raise TypeError(f"Unsupported identifier type for cache key: {type(identifier)}")


def fibonacci_sphere(samples: int, radius: float, center: np.ndarray) -> np.ndarray:
    """Evenly distribute 'samples' points on a sphere of given radius, centered at 'center'."""
    points = np.empty((samples, 3))
    phi = np.pi * (np.sqrt(5.) - 1.)
    denom = float(max(1, samples - 1))
    for i in range(samples):
        y = 1.0 - (i / denom) * 2.0
        r_xy = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points[i] = [np.cos(theta) * r_xy, y, np.sin(theta) * r_xy]
        # ensure finite numbers in case of precision issue
        if not np.all(np.isfinite(points[i])):
            points[i] = np.array([0.0, 1.0 - (i / denom) * 2.0, 0.0])
    return points * radius + center


def calculate_ligand_interactions(
        ion_atoms: Atoms,
        all_ligands: List[Atoms],
        current_lig_idx: int,
        clash_penalty: float = CLASH_PENALTY_FOR_SCORING,
) -> Tuple[float, bool]:
    """
    Compute interaction score for one ligand vs ion and other ligands.
    Returns (score, any_clash).
    """
    eval_lig = all_ligands[current_lig_idx]
    total_repulsion_score = 0.0
    any_clash_detected = False

    # Interaction with ion
    rep_val_ion, clash_ion = calculate_intermolecular_repulsion(ion_atoms, eval_lig)
    if clash_ion:
        any_clash_detected = True
        total_repulsion_score += clash_penalty
    else:
        total_repulsion_score += rep_val_ion

    # Interaction with other ligands
    for i, other_lig in enumerate(all_ligands):
        if i == current_lig_idx:
            continue
        rep_val_pair, clash_pair = calculate_intermolecular_repulsion(eval_lig, other_lig)
        if clash_pair:
            any_clash_detected = True
            total_repulsion_score += clash_penalty
        else:
            total_repulsion_score += rep_val_pair

    return total_repulsion_score, any_clash_detected


def check_system_clashes(ion_atoms: Atoms, all_ligands: List[Atoms]) -> bool:
    """Return True if any pairwise clash is detected in the entire system."""
    if any(calculate_intermolecular_repulsion(ion_atoms, lig)[1] for lig in all_ligands):
        return True
    for i in range(len(all_ligands)):
        for j in range(i + 1, len(all_ligands)):
            if calculate_intermolecular_repulsion(all_ligands[i], all_ligands[j])[1]:
                return True
    return False


# =============================================================================
# Volume estimation
# =============================================================================


def method_convex_hull_volume(atoms: Atoms) -> float:
    """
    Estimate molecular volume using 3D convex hull of atomic positions.
    Robustly handles planar molecules (like NO3-) where ConvexHull fails.
    """
    pts = atoms.get_positions()

    # 1. 点太少，直接返回默认小体积
    if len(pts) < 4:
        return 1.0

    try:
        # 尝试计算凸包体积
        hull = ConvexHull(pts)
        vol = float(hull.volume)
        # 如果算出来体积极其小（接近平面），也给个保底值
        return max(vol, 1.0)

    except QhullError:
        # 2. 捕获 Qhull 错误 (例如平面分子 NO3-)
        # 这种情况下分子虽然几何体积为0，但物理占位不为0
        # 返回一个估算值，例如：原子数 * 这里的经验系数
        # 或者直接返回一个默认值 10.0 (约等于一个小分子的体积)
        return float(len(pts)) * 2.0
    except Exception:
        # 捕获其他可能的异常
        return 5.0


# =============================================================================
# Patch and template helpers
# =============================================================================

def get_ase_and_patch(
        identifier: Union[str, Atoms, Chem.Mol],
        relative_score_threshold: float,
        max_patch_atoms: int,
        verbose: bool,
) -> Tuple[Atoms, List[int]]:
    """
    Normalize various identifiers to (ASE Atoms, patch_indices) using patch_picker.
    """
    if isinstance(identifier, str) and is_xyz_path(identifier):
        ase_obj = read(identifier)
        ase_obj = ase_obj if isinstance(ase_obj, Atoms) else ase_obj[0]
        ase_res, patch_idx = get_patch_atoms_and_indices(
            ase_obj,
            relative_score_threshold=relative_score_threshold,
            max_patch_atoms=max_patch_atoms,
            verbose=verbose,
        )
        return ase_res, patch_idx

    ase_res, patch_idx = get_patch_atoms_and_indices(
        identifier,
        relative_score_threshold=relative_score_threshold,
        max_patch_atoms=max_patch_atoms,
        verbose=verbose,
    )
    return ase_res, patch_idx


def prepare_ion(
        ion_identifier: Union[str, Atoms, Chem.Mol],
        relative_score_threshold: float,
        verbose: bool,
) -> Tuple[Atoms, np.ndarray, str]:
    """
    Build ion as ASE Atoms and return (ion_atoms, ion_center, ion_symbol).
    """
    ion_ase, _ = get_ase_and_patch(
        ion_identifier,
        relative_score_threshold=relative_score_threshold,
        max_patch_atoms=1,
        verbose=verbose,
    )
    ion_center = ion_ase.positions[0] if len(ion_ase) > 0 else np.array([0.0, 0.0, 0.0])
    ion_symbol = ion_ase.get_chemical_symbols()[0] if len(ion_ase) > 0 else "X"
    return ion_ase, ion_center, ion_symbol


def build_ligand_templates(
        ligand_molecule_info: List[Tuple[Union[str, Atoms, Chem.Mol], int]],
        ion_symbol: str,
        relative_score_threshold: float,
        max_patch_atoms: int,
        verbose: bool,
) -> List[Tuple[Atoms, List[int], np.ndarray, float, Union[str, Atoms, Chem.Mol]]]:
    """
    Build ligand templates as tuples.
    Returns a list where identical molecules are grouped together initially.
    """
    cache: Dict[str, Tuple[Atoms, List[int], np.ndarray, float]] = {}
    templates: List[Tuple[Atoms, List[int], np.ndarray, float, Union[str, Atoms, Chem.Mol]]] = []

    if verbose:
        print("\n--- Ligand Template Preparation ---")

    for key_obj, count in ligand_molecule_info:
        cache_key = make_cache_key(key_obj)
        if cache_key not in cache:
            mol_ase, patch_indices = get_ase_and_patch(
                key_obj,
                relative_score_threshold=relative_score_threshold,
                max_patch_atoms=max_patch_atoms,
                verbose=verbose,
            )
            if len(patch_indices) == 0:
                raise ValueError(f"No patch atoms found for ligand: {key_obj}")

            patch_atom_coords = mol_ase.get_positions()[patch_indices]
            patch_centroid_local = np.mean(patch_atom_coords, axis=0)

            ideal_distances = [
                get_bond_length(ion_symbol, mol_ase.get_chemical_symbols()[idx], skin=0)
                for idx in patch_indices
            ]
            avg_ideal_dist = float(np.mean(ideal_distances)) if ideal_distances else get_bond_length(
                ion_symbol, mol_ase.get_chemical_symbols()[0], skin=0
            )

            identifier_name = key_obj if isinstance(key_obj, str) else type(key_obj).__name__
            if verbose:
                print(f"    {identifier_name}: Avg ideal patch-ion distance = {avg_ideal_dist:.3f} Å")

            cache[cache_key] = (mol_ase, patch_indices, patch_centroid_local, avg_ideal_dist)

        mol_ase, patch_indices, centroid, avg_dist = cache[cache_key]
        for _ in range(count):
            templates.append((mol_ase.copy(), list(patch_indices), centroid.copy(), float(avg_dist), key_obj))

    if verbose:
        print("---------------------------------\n")

    return templates


# =============================================================================
# Placement and optimization
# =============================================================================

def place_ligands_initial(
        ion_center: np.ndarray,
        ligand_templates: List[Tuple[Atoms, List[int], np.ndarray, float, Union[str, Atoms, Chem.Mol]]],
        sphere_dist_factor: float,
        orientation_mode: str,
) -> Tuple[List[Atoms], np.ndarray]:
    """
    Place ligands on a sphere around ion.
    The ligand_templates list should be pre-shuffled if random mixing is desired.
    """
    total = len(ligand_templates)
    if total == 0:
        return [], np.zeros((0, 3))

    # volume-aware distance scaling
    volumes = [max(method_convex_hull_volume(tpl[0]), 1e-12) for tpl in ligand_templates]
    vol_ref = float(np.max(volumes)) if len(volumes) > 0 else 1.0

    # Scale factor on length from volume ratio: L ~ V^(1/3)
    volume_scales = []
    for v in volumes:
        raw = (vol_ref / max(v, 1e-12)) ** (1.0 / 8.0)
        volume_scales.append(float(raw))

    base_target_distances = [tpl[3] * sphere_dist_factor for tpl in ligand_templates]
    target_distances = [base_target_distances[i] * volume_scales[i] for i in range(total)]

    directions = fibonacci_sphere(total, radius=1.0, center=np.array([0.0, 0.0, 0.0]))

    ligands: List[Atoms] = []
    target_centroids = np.empty((total, 3))

    for i, (ase_mol, patch_indices, centroid_local, _, _) in enumerate(ligand_templates):
        lig = ase_mol.copy()
        target = ion_center + directions[i] * target_distances[i]
        target_centroids[i] = target

        # Translate so patch centroid is at the target
        lig.translate(target - centroid_local)

        # Orientation
        if orientation_mode == "random" or len(lig) == 1:
            axis = np.random.rand(3) - 0.5
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
            lig.rotate(np.random.uniform(0.0, 360.0), axis, center=target)
        elif orientation_mode == "aligned_to_ion" and patch_indices:
            primary_patch_pos = lig.positions[patch_indices[0]]
            v_centroid_to_patch = primary_patch_pos - target
            v_ion_to_centroid = target - ion_center
            if np.linalg.norm(v_centroid_to_patch) > 1e-8 and np.linalg.norm(v_ion_to_centroid) > 1e-8:
                # Rotate so the primary patch atom points toward the ion
                lig.rotate(v_centroid_to_patch, -v_ion_to_centroid, center=target)

        ligands.append(lig)

    return ligands, target_centroids


def optimize_ligand_orientations(
        ion_atoms: Atoms,
        ligands: List[Atoms],
        rotation_centers: np.ndarray,
        rotation_opt_iterations: int,
        rotation_samples_per_ligand: int,
        verbose: bool,
        debug_save_dir: str = None,
        current_sphere_attempt: int = 0
) -> List[Atoms]:
    """
    Stochastic orientation optimization by random local rotations around each ligand’s patch centroid.
    """
    total = len(ligands)
    for rot_iter in range(rotation_opt_iterations):
        order = list(range(total))
        random.shuffle(order)
        improvements = 0

        for idx in order:
            current = ligands[idx]
            center = rotation_centers[idx]
            base_score, _ = calculate_ligand_interactions(
                ion_atoms, ligands, idx, clash_penalty=CLASH_PENALTY_FOR_SCORING
            )

            best_local = current.copy()
            best_score = base_score

            for _ in range(rotation_samples_per_ligand):
                trial = current.copy()
                axis = np.random.rand(3) - 0.5
                norm = np.linalg.norm(axis)
                axis = axis / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
                trial.rotate(np.random.uniform(0.0, 360.0), axis, center=center)

                tmp = list(ligands)
                tmp[idx] = trial
                trial_score, _ = calculate_ligand_interactions(
                    ion_atoms, tmp, idx, clash_penalty=CLASH_PENALTY_FOR_SCORING
                )

                if trial_score < best_score:
                    best_score = trial_score
                    best_local = trial.copy()

            if best_score < base_score - 1e-6:
                improvements += 1
            ligands[idx] = best_local

        if verbose and debug_save_dir and (rot_iter + 1) % 3 == 0:
            step_snapshot = ion_atoms.copy()
            for lig in ligands:
                step_snapshot.extend(lig)
            step_filename = os.path.join(debug_save_dir, f"step_{current_sphere_attempt + 1}_rot_{rot_iter + 1}.xyz")
            write(step_filename, step_snapshot)

        if verbose and (rot_iter % 10 == 0 or rot_iter == rotation_opt_iterations - 1 or improvements == 0):
            rep_sum_no_clash = 0.0
            num_clashing = 0
            for j in range(total):
                sc, cl = calculate_ligand_interactions(
                    ion_atoms, ligands, j, clash_penalty=CLASH_PENALTY_FOR_SCORING
                )
                if cl:
                    num_clashing += 1
                else:
                    rep_sum_no_clash += sc
            print(
                f"    Rot.Opt Iter {rot_iter + 1}: Rep Sum (non-clashing): {rep_sum_no_clash:.2f}, "
                f"Ligands with clashes: {num_clashing}, Improvements: {improvements}"
            )

        if improvements == 0 and rot_iter > min(5, rotation_opt_iterations // 3):
            if verbose:
                print("    Rotation optimization converged early.")
            break

    return ligands


def evaluate_configuration(
        ion_atoms: Atoms,
        ligands: List[Atoms],
) -> Tuple[bool, float]:
    """
    Return (has_clashes, total_system_score_with_penalties).
    """
    has_clashes = check_system_clashes(ion_atoms, ligands)
    total_score = 0.0
    for k in range(len(ligands)):
        sc, _ = calculate_ligand_interactions(
            ion_atoms, ligands, k, clash_penalty=CLASH_PENALTY_FOR_SCORING
        )
        total_score += sc
    return has_clashes, total_score


# =============================================================================
# Main cluster builder
# =============================================================================

def build_cluster(
        ion_identifier: Union[str, Atoms, Chem.Mol],
        ligand_molecule_info: List[Tuple[Union[str, Atoms, Chem.Mol], int]],
        relative_score_threshold: float = 0.85,
        max_patch_atoms: int = 3,
        initial_sphere_skin_factor: float = 0.75,
        sphere_skin_increment_factor: float = 0.02,
        max_sphere_expansions: int = 20,
        target_no_clashes: bool = True,
        rotation_opt_iterations: int = 50,
        rotation_samples_per_ligand: int = 80,
        initial_ligand_orientation: str = "aligned_to_ion",
        verbose: bool = True,
        debug_save_dir: str = None,
) -> Atoms:
    """
    Build a cluster: ion + multiple ligands.
    """
    if verbose:
        total_ligs = sum(count for _, count in ligand_molecule_info)
        print(f"--- Cluster Build Initiated: {str(ion_identifier)} + {total_ligs} Ligands ---")
        print(f"  Using DEFAULT_CLASH_FACTOR (from CombineMols3D): {DEFAULT_CLASH_FACTOR}")
        if debug_save_dir:
            print(f"  Debug output enabled. Saving intermediate steps to: {debug_save_dir}")

    if verbose and debug_save_dir:
        if not os.path.exists(debug_save_dir):
            os.makedirs(debug_save_dir, exist_ok=True)

    ase_ion, ion_center, ion_symbol = prepare_ion(
        ion_identifier=ion_identifier,
        relative_score_threshold=relative_score_threshold,
        verbose=verbose,
    )

    ligand_templates = build_ligand_templates(
        ligand_molecule_info=ligand_molecule_info,
        ion_symbol=ion_symbol,
        relative_score_threshold=relative_score_threshold,
        max_patch_atoms=max_patch_atoms,
        verbose=verbose,
    )
    if len(ligand_templates) == 0:
        return ase_ion

    # =========================================================================
    # FIX: Random shuffle the templates to mix different species on the sphere
    # =========================================================================
    # Before this, the list is [TypeA, TypeA, ..., TypeB, TypeB].
    # Since sphere points are generated in spiral order, this caused clustering.
    random.shuffle(ligand_templates)
    # =========================================================================

    best_config: List[Atoms] = []
    best_score = float("inf")
    best_has_clash = True

    sphere_factor = initial_sphere_skin_factor
    for attempt in range(max_sphere_expansions):
        if verbose:
            print(
                f"--- Sphere Expansion Attempt {attempt + 1}/{max_sphere_expansions} (Factor: {sphere_factor:.3f}) ---")

        # Initial placement (volume-aware)
        placed, centers = place_ligands_initial(
            ion_center=ion_center,
            ligand_templates=ligand_templates,
            sphere_dist_factor=sphere_factor,
            orientation_mode=initial_ligand_orientation,
        )

        # Orientation optimization
        placed = optimize_ligand_orientations(
            ion_atoms=ase_ion,
            ligands=placed,
            rotation_centers=centers,
            rotation_opt_iterations=rotation_opt_iterations,
            rotation_samples_per_ligand=rotation_samples_per_ligand,
            verbose=verbose,
            debug_save_dir=debug_save_dir,
            current_sphere_attempt=attempt
        )

        # Evaluate
        has_clashes, total_score = evaluate_configuration(ase_ion, placed)
        if verbose:
            status = "Clashes Present" if has_clashes else "Clash-Free"
            print(f"  End Sphere Attempt {attempt + 1}: Status = {status}, Total Score = {total_score:.2f}")

        # Track best
        improved = (
                best_config == []
                or (not has_clashes and best_has_clash)
                or (not has_clashes and not best_has_clash and total_score < best_score)
                or (has_clashes and best_has_clash and not target_no_clashes and total_score < best_score)
        )
        if improved:
            best_config = [lig.copy() for lig in placed]
            best_score = total_score
            best_has_clash = has_clashes
            if verbose:
                print(f"    New best configuration. Score: {best_score:.2f}, Clashes: {best_has_clash}")

        if target_no_clashes and not best_has_clash:
            if verbose:
                print("  SUCCESS: Clash-free configuration found. Stopping sphere expansion.")
            break

        sphere_factor += sphere_skin_increment_factor

    final_cluster = ase_ion.copy()
    for lig in best_config:
        final_cluster.extend(lig)

    if verbose:
        status = "CLASH-FREE" if not best_has_clash else "WITH CLASHES (penalized in score)"
        print(f"\n--- Final Cluster Configuration ({status}) ---")
        print(f"  Total Atoms: {len(final_cluster)}, Final Score: {best_score:.2f}")

    return final_cluster


# =============================================================================
# Examples
# =============================================================================
if __name__ == "__main__":
    from ase.build import molecule

    # Example setup
    dme = "COCCOC"  # DME
    water_atoms = molecule("H2O")

    # 比如这里输入 2个DME 和 2个H2O
    # 之前可能出现 [DME, DME, H2O, H2O] 的顺序分布在球面上
    # 现在会随机混合，比如 [DME, H2O, DME, H2O] 或其他随机排列
    debug_dir = "debug_mixed_steps"

    cluster_mixed = build_cluster(
        ion_identifier="Li",
        ligand_molecule_info=[(dme, 2), (water_atoms, 2)],
        relative_score_threshold=0.7,
        max_patch_atoms=2,
        target_no_clashes=True,
        verbose=True,
        debug_save_dir=debug_dir
    )
    write("Li_2DME_2H2O_mixed.xyz", cluster_mixed)
    print("Wrote Li_2DME_2H2O_mixed.xyz")