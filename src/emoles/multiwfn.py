import os
import re
from pathlib import Path


class MultiwfnRunner:
    """
    Base class for running Multiwfn calculations.
    Provides common functionality for running Multiwfn with various commands.
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

        # Check if input file exists
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
        # Create command and output filenames based on input file and class name
        base_name = Path(self.filename).stem
        class_name = self.__class__.__name__
        command_filename = f"{base_name}_{class_name}_command.txt"
        output_filename = f"{base_name}_{class_name}_output.txt"

        # Write commands to file
        with open(command_filename, 'w') as command_file:
            # Write each command on a separate line
            command_file.write('\n'.join(commands))

        # Execute Multiwfn with input redirection
        command = f"Multiwfn {self.input_path} < {command_filename} > {output_filename}"
        print(f"Executing: {command}")

        # Run the command as a shell command
        os.system(command)
        # Read output from file
        with open(output_filename, 'r') as output_file:
            output = output_file.read()

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

    def calculate_grid_data(self):
        """
        Calculate ESP grid data and export to cube file.

        Returns:
        --------
        dict
            Dictionary containing ESP maximum and minimum values and their locations
        str
            Path to the generated cube file
        """
        print(f"Calculating ESP grid data for {self.filename} ...")

        # Commands for ESP grid calculation:
        # 5: Output and plot specific property within a spatial region (calc. grid data)
        # 12: Total electrostatic potential (ESP)
        # 2: Medium quality grid, covering whole system
        # 2: Export data to a Gaussian-type cube file
        # q: Quit
        commands = ["5", "12", "2", "2", "q"]

        # Run Multiwfn with ESP grid commands
        output = self.run_multiwfn(commands)

        # Parse the output to extract ESP extrema information
        esp_data = self._parse_esp_output(output)
        output_cube = f"totesp.cub"
        cube_path = os.path.join(self.directory, output_cube)

        return esp_data, cube_path

    def _parse_esp_output(self, output):
        """
        Parse Multiwfn output to extract ESP extrema information.

        Parameters:
        -----------
        output : str
            Multiwfn output text

        Returns:
        --------
        dict
            Dictionary containing ESP maximum and minimum values and their locations
        """
        # Initialize results dictionary
        esp_data = {
            'max_value': None,
            'max_location': None,
            'min_value': None,
            'min_location': None
        }

        # Regular expressions to match the ESP data
        min_pattern = r"The minimum is\s+(\S+)\s+at\s+(\S+)\s+(\S+)\s+(\S+)\s+Bohr"
        max_pattern = r"The maximum is\s+(\S+)\s+at\s+(\S+)\s+(\S+)\s+(\S+)\s+Bohr"

        # Search for matches
        min_match = re.search(min_pattern, output)
        max_match = re.search(max_pattern, output)

        # Extract values if matches found
        if min_match:
            esp_data['min_value'] = float(min_match.group(1))
            esp_data['min_location'] = (
                float(min_match.group(2)),
                float(min_match.group(3)),
                float(min_match.group(4))
            )

        if max_match:
            esp_data['max_value'] = float(max_match.group(1))
            esp_data['max_location'] = (
                float(max_match.group(2)),
                float(max_match.group(3)),
                float(max_match.group(4))
            )

        return esp_data


# Test the classes with test.fch in current directory
if __name__ == "__main__":
    # Test the ESP Calculator with updated commands
    print("\n=== Testing ESP Calculator ===")
    esp_calculator = ESPCalculator("test.fch")
    esp_results, cube_file = esp_calculator.calculate_grid_data()
    print("\nESP Grid Data Results:")
    print(f"ESP cube file created: {cube_file}")
    print(f"Maximum ESP: {esp_results['max_value']:.6f} a.u. at {esp_results['max_location']}")
    print(f"Minimum ESP: {esp_results['min_value']:.6f} a.u. at {esp_results['min_location']}")