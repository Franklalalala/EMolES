import os
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from ase.data import chemical_symbols  # ASE interface for element data
from emoles.fchk_parser import read_fchk


# ========================= PDB Output Functions =========================
def save_points_to_pdb(points, filename):
    """
    Save point cloud to a simplified PDB file.
    (All points are masqueraded as C atoms for visualization in VMD/ChimeraX).

    Args:
        points: (N,3) numpy array, units in Å.
        filename: Output filename.
    """
    if len(points) == 0:
        return
    with open(filename, 'w') as f:
        for i, (x, y, z) in enumerate(points):
            # "PTS" is used as the residue name for points
            f.write(
                f"ATOM  {i + 1:>5}  X   PTS A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")
    print(f"   [Save] Surface points saved to: {filename}")


def save_molecule_to_pdb(atomic_numbers, coords, filename="molecule.pdb"):
    """
    Save the geometry of the entire molecule to a simple PDB.
    - All atoms are assigned to the same residue 'MOL'.
    - Element symbols are derived via ASE.
    """
    if len(coords) == 0:
        return

    with open(filename, "w") as f:
        for i, (Z, (x, y, z)) in enumerate(zip(atomic_numbers, coords)):
            # Use ASE's chemical_symbols (index 0 is X, 1 is H, etc.)
            elem = chemical_symbols[int(Z)]
            name = elem
            resName = "MOL"
            chainID = "A"
            resSeq = 1
            serial = i + 1

            line = (
                f"ATOM  {serial:5d} {name:<4s}{resName:>3s} {chainID}"
                f"{resSeq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {elem:>2s}\n"
            )
            f.write(line)
        f.write("END\n")

    print(f"[Save] Molecule coordinates saved to: {filename}")

# ========================= Geometry & Convex Hull Helpers =========================
def is_point_in_hull(point, hull, tol=1e-4):
    """
    Check if a point is inside (or on the surface of) a ConvexHull.
    The hull equations are: A*x + b <= 0 for interior points.
    """
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    return np.all(A.dot(point) + b <= tol)


def extract_li_envelope(points, li_center):
    """
    Extract the specific connected component of the point cloud that envelops the Li atom.

    Logic:
    1) Use KDTree + clustering radius (r) to separate points into connected components.
    2) Calculate ConvexHull for each component.
    3) Select the component whose hull contains the Li center.
    4) If multiple satisfy this, pick the largest.
    5) Fallback: If no hull contains Li, pick the component with the centroid closest to Li.

    Returns:
        (M,3) numpy array of selected points.
    """
    n_points = len(points)
    if n_points < 4:
        return None

    # Determine clustering radius based on nearest neighbor distances
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  # dists[:,1] is the distance to the nearest neighbor
    nn_dist = np.median(dists[:, 1])

    if not np.isfinite(nn_dist) or nn_dist <= 0:
        return points

    r = nn_dist * 1.5
    neighbors = tree.query_ball_point(points, r)

    # Perform clustering (Breadth-First Search)
    labels = -np.ones(n_points, dtype=int)
    comp_id = 0
    for i in range(n_points):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = comp_id
        while stack:
            j = stack.pop()
            for k in neighbors[j]:
                if labels[k] == -1:
                    labels[k] = comp_id
                    stack.append(k)
        comp_id += 1

    # Strategy 1: Find cluster whose Convex Hull contains Li
    best_cluster_points = None
    best_cluster_size = -1
    for cid in range(comp_id):
        idx = np.where(labels == cid)[0]
        pts = points[idx]
        if len(pts) < 4:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Coplanar points or other Qhull errors
            continue

        if is_point_in_hull(li_center, hull):
            if len(pts) > best_cluster_size:
                best_cluster_size = len(pts)
                best_cluster_points = pts

    if best_cluster_points is not None:
        return best_cluster_points

    # Strategy 2: Fallback to cluster with centroid closest to Li
    best_cluster_points = None
    best_dist = None
    for cid in range(comp_id):
        idx = np.where(labels == cid)[0]
        pts = points[idx]
        if len(pts) < 4:
            continue
        centroid = pts.mean(axis=0)
        d = np.linalg.norm(centroid - li_center)
        if (best_dist is None) or (d < best_dist):
            best_dist = d
            best_cluster_points = pts

    return best_cluster_points

# ========================= Multiwfn Wrapper =========================
class MultiwfnRunner:
    """
    Base class for automating Multiwfn calculations via command line redirection.
    """

    def __init__(self, filename, directory='.'):
        """
        Initialize the MultiwfnRunner with a file to analyze.

        Parameters:
        -----------
        filename : str
            The name of the file to process
        directory : str, optional
            The directory containing the input file (default: current directory)
        """
        self.filename = filename
        self.directory = os.path.abspath(directory)
        self.input_path = os.path.join(directory, filename)

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def run_multiwfn(self, commands):
        """
        Run Multiwfn with the specified commands.

        Parameters:
        -----------
        commands : list
            List of commands to pass to Multiwfn, one command per element

        Returns:
        --------
        str
            Output from the Multiwfn process
        """
        base_name = Path(self.filename).stem
        class_name = self.__class__.__name__
        command_filename = f"{base_name}_{class_name}_cmd.txt"
        output_filename = f"{base_name}_{class_name}_out.txt"

        # Write commands to a temporary file
        with open(command_filename, 'w') as command_file:
            command_file.write('\n'.join(commands))

        # Execute Multiwfn with input redirection
        command = (f"Multiwfn {self.input_path} < {command_filename} > {output_filename}")
        print(f"Executing: {command}")

        ret_code = os.system(command)
        if ret_code != 0:
            print(f"  [Warning] Multiwfn exited with code {ret_code}")

        # Read output
        try:
            with open(output_filename, 'r', encoding='utf-8', errors='ignore') as output_file:
                output = output_file.read()
        except FileNotFoundError:
            output = ""
            print("  [Error] Output file not found.")

        return output


class RESPChargeCalculator(MultiwfnRunner):
    """Specialized class for RESP charge calculations"""

    def calculate(self):
        """
        Calculate RESP charges using Multiwfn.

        Returns:
        --------
        str
            Path to the generated charge file
        """
        print(f"Calculating RESP charge for {self.filename} ...")

        # Commands for RESP charge calculation
        commands = ["7", "18", "1", "y", "0", "0", "q"]

        # Run Multiwfn with RESP charge commands
        output = self.run_multiwfn(commands)

        # Generate output charge filename
        base_name = Path(self.filename).stem
        chg_file = f"{base_name}.chg"
        chg_file_path = os.path.join(self.directory, chg_file)

        # Check if charge file was created
        if not os.path.exists(chg_file_path):
            raise FileNotFoundError(f"Expected charge file was not created: {chg_file_path}")

        return chg_file_path


class ESPCalculator(MultiwfnRunner):
    """Specialized class for ESP (Electrostatic Potential) analysis"""

    def get_ESP_value(self):
        """
        Calculate ESP surface extrema.

        Returns:
        --------
        dict
            Dictionary containing ESP maximum and minimum values (in eV) and their locations
        """
        print(f"Calculating ESP surface extrema for {self.filename} ...")

        # Commands for ESP surface extrema calculation:
        # 12: Analyze surface electrostatic potential extremum
        # 0: Use default electron density isosurface
        # q: Quit
        commands = ["12", "0", "q"]

        # Run Multiwfn
        output = self.run_multiwfn(commands)

        # Parse the output to extract ESP extrema information
        esp_data = self._parse_esp_output(output)

        return esp_data

    def _parse_esp_output(self, output):
        """
        Parse Multiwfn output to extract ESP surface extrema information.
        The expected output format is:
         Global surface minimum: -0.063895 a.u. at  -0.073819  -1.492493   0.073205 Ang
         Global surface maximum:  0.017945 a.u. at   0.073580   2.371441   1.030127 Ang

        Parameters:
        -----------
        output : str
            Multiwfn output text

        Returns:
        --------
        dict
            Dictionary containing ESP maximum and minimum values and their locations
        """
        # Conversion factor from Hartree (a.u.) to eV
        HARTREE_TO_EV = 27.2114

        esp_data = {
            'ESP_max_eV': None,
            'ESP_max_location_Ang': None,
            'ESP_min_eV': None,
            'ESP_min_location_Ang': None
        }

        # Regular expressions to match the ESP data
        min_pattern = r"Global surface minimum:\s+([\-0-9.]+)\s+a\.u\.\s+at\s+([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+Ang"
        max_pattern = r"Global surface maximum:\s+([\-0-9.]+)\s+a\.u\.\s+at\s+([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+Ang"

        min_match = re.search(min_pattern, output)
        max_match = re.search(max_pattern, output)

        if min_match:
            # Extract value in a.u. and convert to eV
            esp_min_au = float(min_match.group(1))
            esp_data['ESP_min_eV'] = esp_min_au * HARTREE_TO_EV
            # Extract location in Angstroms
            esp_data['ESP_min_location_Ang'] = (
                float(min_match.group(2)),
                float(min_match.group(3)),
                float(min_match.group(4))
            )

        if max_match:
            # Extract value in a.u. and convert to eV
            esp_max_au = float(max_match.group(1))
            esp_data['ESP_max_eV'] = esp_max_au * HARTREE_TO_EV
            # Extract location in Angstroms
            esp_data['ESP_max_location_Ang'] = (
                float(max_match.group(2)),
                float(max_match.group(3)),
                float(max_match.group(4))
            )

        return esp_data

    def get_acc_grid_data(self):
        """
        Calculate and export high-quality electron density and ESP grid data.
        The ESP grid data is converted to eV.

        Returns:
        --------
        dict
            A dictionary with paths to the generated cube files:
            {'density_cub': path_to_density_cub, 'esp_cub': path_to_esp_cub}
        """
        print(f"Calculating high-quality grid data for {self.filename} ...")

        # Commands for generating density and ESP grids, and converting ESP to eV
        commands = [
            "5",  # Main function: Calculate grid data
            "1",  # Property: Electron density
            "3",  # Grid quality: High
            "2",  # Action: Export to cube file (creates density.cub)
            "0",  # Return to property selection menu
            "5",  # Main function: Calculate grid data
            "12", # Property: Total electrostatic potential (ESP)
            "1",  # Grid quality: Low (as per request)
            "2",  # Action: Export to cube file (creates totesp.cub)
            "q",  # Quit
        ]

        # Run Multiwfn
        self.run_multiwfn(commands)

        # Rename output files
        base_name = Path(self.filename).stem
        class_name = self.__class__.__name__

        # Define original and new paths
        old_density_path = os.path.join(self.directory, 'density.cub')
        new_density_path = os.path.join(self.directory, f"{base_name}_{class_name}_density.cub")

        old_totesp_path = os.path.join(self.directory, 'totesp.cub')
        new_totesp_path = os.path.join(self.directory, f"{base_name}_{class_name}_totesp.cub")

        # Perform renaming and cleanup
        if os.path.exists(old_density_path):
            if os.path.exists(new_density_path):
                os.remove(new_density_path)
            os.rename(old_density_path, new_density_path)
        else:
            print(f"Warning: Expected file {old_density_path} not found.")

        if os.path.exists(old_totesp_path):
            if os.path.exists(new_totesp_path):
                os.remove(new_totesp_path)
            os.rename(old_totesp_path, new_totesp_path)
        else:
            print(f"Warning: Expected file {old_totesp_path} not found.")

        return {
            'density_cub': new_density_path,
            'esp_cub': new_totesp_path
        }

# ========================= ELF Deformation Calculator =========================
class ELFDeformationCalculator(MultiwfnRunner):
    """
    Calculates the Deformation Factor (phi) of the ELF isosurface around a Li atom.
    Workflow: Multiwfn grid generation -> Filter points -> Convex Hull analysis.
    """

    def __init__(self, filename, directory='.',
                 isovalue=0.5, diff_list=None, li_cutoff=1.1):
        """
        Initialize calculator and set parameters.

        Args:
            filename: Target file (e.g., .fch).
            directory: Working directory.
            isovalue: Center value for ELF isosurfaces (default: 0.5).
            diff_list: List of widths (diff) to analyze (default: [0.09]).
            li_cutoff: Spatial screening radius around Li center in Å (default: 1.1).
        """
        super().__init__(filename, directory)
        self.isovalue = isovalue
        self.diff_list = diff_list if diff_list is not None else [0.09]
        self.li_cutoff = li_cutoff

    def calculate(
            self,
            atom_index_1based,
            li_center,
            li_id=None,
            radius=3.0,
            grid_spacing=0.1,
    ):
        """
        Parameters
        ----------
        atom_index_1based : int
            1-based index of Li atom in Multiwfn.
        li_center : array-like
            Coordinates of Li atom (Å).
        li_id : int or None
            Identifier for naming output PDBs.
        radius : float
            Extension distance for grid calculation (Bohr).
        grid_spacing : float
            Grid spacing (Bohr).

        Returns
        -------
        dict
            Results containing volume, area, phi, and interpretation.
        """
        # Multiwfn commands:
        # 5: Output property data
        # 9: Grid data
        # 11: Grid data within sphere around atom
        # ... params ...
        # 4: Export data
        # ... isovalue setup ...
        # 3: Export to plain text (output.txt)
        # q: Quit
        commands = [
            "5",
            "9",
            "11",
            str(atom_index_1based),
            str(radius),
            str(grid_spacing),
            "4",
            str(self.isovalue),
            "3",
            "q",
        ]

        self.run_multiwfn(commands)

        return self._parse_output(
            li_center=np.asarray(li_center),
            li_id=li_id,
        )

    def _parse_output(self, li_center, li_id=None):
        """
        Parses 'output.txt' from Multiwfn, filters points by ELF value and space,
        extracts the Li envelope, and computes geometric properties.
        """
        results = {}

        if not os.path.exists("output.txt"):
            print("  [Error] output.txt not found.")
            return results

        # Read whitespace-separated values
        data = pd.read_csv(
            "output.txt",
            header=None,
            sep=r'\s+',
            names=['x', 'y', 'z', 'value']
        )

        print(f"   Points raw: {len(data)}")

        for diff in self.diff_list:
            print(f"-> Processing Li center at {li_center} with DIFF={diff}...")

            # 1. Filter by ELF value (Isosurface Band)
            # Use self.isovalue
            iso = data[
                (data['value'] > self.isovalue - diff) &
                (data['value'] < self.isovalue + diff)
                ].reset_index(drop=True)

            if len(iso) == 0:
                print(f"    [Warning] No points found within ELF threshold for DIFF={diff}.")
                continue

            xyz = iso[['x', 'y', 'z']].to_numpy()

            # 2. Filter by distance to Li (Spatial Screening)
            # Use self.li_cutoff
            dist_li = np.linalg.norm(xyz - li_center, axis=1)
            keep_mask = dist_li < self.li_cutoff
            filtered_points = xyz[keep_mask]

            print(
                f"   Points raw: {len(data)} -> Isovalue filtered: {len(iso)} -> "
                f"Spatial filtered: {len(filtered_points)}"
            )

            if len(filtered_points) < 4:
                print(f"    [Error] Not enough points to form a convex hull after screening.")
                continue

            # 3. Connectivity filtering: Keep only the envelope surrounding Li
            li_surface_points = extract_li_envelope(filtered_points, li_center)

            if li_surface_points is None or len(li_surface_points) < 4:
                print(f"    [Error] Unable to identify a continuous Li ELF surface around this Li.")
                continue

            print(f"   After connectivity filtering (Li envelope): {len(li_surface_points)} points")

            # 4. Save surface points to PDB (if ID provided)
            if li_id is not None:
                diff_tag = str(diff).replace('.', 'p')
                pdb_name = f"Li_{li_id}_diff{diff_tag}_surface.pdb"
                save_points_to_pdb(li_surface_points, pdb_name)

            # 5. Compute Volume, Area, and Deformation Factor (Phi) via Convex Hull
            try:
                hull = ConvexHull(li_surface_points)
                vol = hull.volume  # Å^3
                area = hull.area  # Å^2

                # Calculate radius of equivalent sphere
                radius_eq = (vol / (4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
                area_sphere = 4.0 * math.pi * radius_eq ** 2

                # Phi = Actual Area / Area of Equivalent Sphere
                phi = area / area_sphere

                # Interpretation threshold (1.015)
                if phi > 1.015:
                    interpretation = "Li-Bond (Deformed)"
                else:
                    interpretation = "Ionic-Bond (Spherical)"

                results[diff] = {
                    "volume_ang3": vol,
                    "area_ang2": area,
                    "phi": phi,
                    "interpretation": interpretation,
                    "radius_eq_sphere": radius_eq,
                    "area_eq_sphere": area_sphere,
                }

            except Exception as e:
                print(f"    [Error] ConvexHull calculation failed: {e}")
                continue

        return results


# ========================= Main Execution =========================

def main():
    target_file = 'target.fch'

    if not os.path.exists(target_file):
        print(f"Error: File '{target_file}' not found.")
        return

    esp_calculator = ESPCalculator("target.fch")

    # Test the ESP Calculator for ESP surface extrema
    print("\n=== Testing ESPCalculator.get_ESP_value ===")
    esp_results = esp_calculator.get_ESP_value()
    print("\nESP Surface Extrema Results:")
    if esp_results.get('ESP_max_eV') is not None:
        print(f"Maximum ESP: {esp_results['ESP_max_eV']:.6f} eV at {esp_results['ESP_max_location_Ang']}")
        print(f"Minimum ESP: {esp_results['ESP_min_eV']:.6f} eV at {esp_results['ESP_min_location_Ang']}")
    else:
        print("Could not parse ESP extrema from output.")

    # Test the ESP Calculator for generating grid data
    print("\n=== Testing ESPCalculator.get_acc_grid_data ===")
    grid_files = esp_calculator.get_acc_grid_data()
    print("\nGrid Data Generation Results:")
    print(f"Density cube file created: {grid_files['density_cub']}")
    print(f"ESP cube file created: {grid_files['esp_cub']}")

    print(f"--- Analyzing {target_file} ---")
    print("Reading FCHK file to identify atoms...")

    # emoles.fchk_parser.read_fchk returns an ASE Atoms object
    atoms = read_fchk(target_file)

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()  # Å
    atomic_numbers = atoms.get_atomic_numbers()

    # Save the whole molecule PDB for visualization context
    save_molecule_to_pdb(atomic_numbers, positions, filename="molecule.pdb")

    # Find Li atom indices (ASE uses 0-based indexing)
    li_indices_0based = [i for i, s in enumerate(symbols) if s == 'Li']
    if not li_indices_0based:
        print("Error: No Lithium (Li) atoms found in the system!")
        return

    print(f"Found {len(li_indices_0based)} Li atom(s) at index (0-based): {li_indices_0based}")

    # Initialize Calculator with desired parameters
    calculator = ELFDeformationCalculator(
        target_file,
        isovalue=0.5,
        diff_list=[0.09],
        li_cutoff=1.1
    )

    for i_0based in li_indices_0based:
        i_1based = i_0based + 1
        li_center = positions[i_0based]

        print(f"\nProcessing Li atom #{i_1based}...")

        results = calculator.calculate(
            atom_index_1based=i_1based,
            li_center=li_center,
            li_id=i_1based,
            radius=3.0,  # Bohr
            grid_spacing=0.1  # Bohr
        )

        if not results:
            print(f"  > Calculation failed or insufficient points for Li #{i_1based}")
            continue

        print(f"  > Li Index (Multiwfn): {i_1based}")

        # Iterate over the diff_list stored in the calculator instance
        for diff in calculator.diff_list:
            res = results.get(diff)
            if res is None:
                print(f"    DIFF = {diff:.3f}: insufficient points / skipped")
                continue

            print("-" * 50)
            print(f">> Diff: {diff}")
            print("-" * 50)
            print(f"{'Metric':<30} | {'Value':<20}")
            print("-" * 50)
            print(f"{'ELF Volume (A^3)':<30} | {res['volume_ang3']:.5f}")
            print(f"{'ELF Area (A^2)':<30} | {res['area_ang2']:.5f}")
            print(f"{'Eq. Sphere Radius (A)':<30} | {res['radius_eq_sphere']:.5f}")
            print(f"{'Eq. Sphere Area (A^2)':<30} | {res['area_eq_sphere']:.5f}")
            print("-" * 50)
            print(f"{'Deformation Factor (phi)':<30} | {res['phi']:.5f}")
            print("-" * 50)
            print(f">> Conclusion: {res['interpretation']}")


if __name__ == "__main__":
    main()