import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import warnings
import logging
import sys
from datetime import datetime
import contextlib

# Comprehensive warning suppression for Intel MKL
os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['MKL_VERBOSE'] = '0'
os.environ['BLIS_NUM_THREADS'] = '1'

# Suppress all warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress logging warnings
logging.getLogger().setLevel(logging.ERROR)


class LogCapture:
    """Context manager to capture all stdout and stderr to file"""

    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.error_terminal = sys.stderr

    def __enter__(self):
        self.log = open(self.log_file, 'w')
        sys.stdout = self.log
        sys.stderr = self.log
        return self

    def __exit__(self, typ, val, tb):
        sys.stdout = self.terminal
        sys.stderr = self.error_terminal
        self.log.close()


def setup_logging(log_file=None):
    """Setup logging configuration"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"causality_simulation_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


def print_to_both(message, log_file):
    """Print to both console and log file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')


try:
    from pyedm_ccm_fixed import run_ccm_simulations, plot_ccm_results
except ImportError:
    print("Please make sure pyedm_ccm_fixed.py is in your directory")
    exit(1)


def generate_simulation_1(n_points=1000, strength=0.7, delay=3, noise_level=0.1, coupling=0.8, seed=42):
    """
    Simulation 1: Linear chain - 1 → 2 → 3 → 4
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(4):
        x1[i] = np.random.random()
        x2[i] = np.random.random()
        x3[i] = np.random.random()
        x4[i] = np.random.random()

    # Generate the linear chain
    for i in range(4, n_points):
        # X1 is the driving force
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X2 is influenced by X1
        x2[i] = strength * x1[i - delay] + coupling * x2[i - 1] + noise_level * np.random.random()

        # X3 is influenced by X2
        x3[i] = strength * x2[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()

        # X4 is influenced by X3
        x4[i] = strength * x3[i - delay] + coupling * x4[i - 1] + noise_level * np.random.random()

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def generate_simulation_2(n_points=1000, strong_strength=0.8, weak_strength=0.3, delay=3, noise_level=0.1, coupling=0.8,
                          seed=42):
    """
    Simulation 2: 1 → 2 (strong), 1 → 3 (weak), 4 is independent
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(4):
        x1[i] = np.random.random()
        x2[i] = np.random.random()
        x3[i] = np.random.random()
        x4[i] = np.random.random()

    # Generate the hub structure
    for i in range(4, n_points):
        # X1 is the driving force
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X2 is strongly influenced by X1
        x2[i] = strong_strength * x1[i - delay] + coupling * x2[i - 1] + noise_level * np.random.random()

        # X3 is weakly influenced by X1
        x3[i] = weak_strength * x1[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()

        # X4 is independent
        x4[i] = np.cos(0.12 * i) + coupling * x4[i - 1] + noise_level * np.random.random()

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def generate_simulation_3(n_points=1000, strong_strength=0.8, weak_strength=0.3, delay=3, noise_level=0.1, coupling=0.8,
                          seed=42):
    """
    Simulation 3: 1 → 2 (strong), 1 → 3 (weak), 4 → 2 (weak)
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(4):
        x1[i] = np.random.random()
        x2[i] = np.random.random()
        x3[i] = np.random.random()
        x4[i] = np.random.random()

    # Generate the network structure
    for i in range(4, n_points):
        # X1 is the driving force
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X4 is independent (needs to be generated before X2 since X4 influences X2)
        x4[i] = np.cos(0.12 * i) + coupling * x4[i - 1] + noise_level * np.random.random()

        # X2 is influenced by both X1 (strong) and X4 (weak)
        x2[i] = (strong_strength * x1[i - delay] +
                 weak_strength * x4[i - delay] +
                 coupling * x2[i - 1] +
                 noise_level * np.random.random())

        # X3 is weakly influenced by X1
        x3[i] = weak_strength * x1[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def analyze_all_pairs(data_dict, num_simulations=15, lib_sizes=None):
    """Analyze all pairwise relationships using CCM."""
    if lib_sizes is None:
        lib_sizes = [50, 100, 150, 200, 250]

    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    # Analyze all pairwise relationships
    for var1, var2 in combinations(variables, 2):
        print(f"  Analyzing {var1} ↔ {var2}...")

        result = run_ccm_simulations(
            time_series_1=data_dict[var1],
            time_series_2=data_dict[var2],
            num_simulations=num_simulations,
            lib_sizes=lib_sizes,
            verbose=False
        )

        results[f"{var1}_{var2}"] = result

    return results


def create_causality_plot(results, title, expected_relationships=None, save_path=None):
    """Create a causality matrix plot."""
    variables = ['X1', 'X2', 'X3', 'X4']
    n_vars = len(variables)

    # Create matrix for causality
    causality_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results and results[key]:
                    result = results[key]
                    max_lib = max(result['lib_sizes'])

                    if var1 < var2:  # Original order
                        if max_lib in result['X_predicts_Y']:
                            causality_matrix[i, j] = result['X_predicts_Y'][max_lib]['mean']
                    else:  # Swapped order
                        if max_lib in result['Y_predicts_X']:
                            causality_matrix[i, j] = result['Y_predicts_X'][max_lib]['mean']

    # Plot the causality matrix
    plt.figure(figsize=(10, 8))

    # Create heatmap
    im = plt.imshow(causality_matrix, cmap='RdBu_r', vmin=-0.5, vmax=1)
    plt.colorbar(im, label='CCM ρ')

    # Set labels
    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nCausality Matrix (Row → Column)')

    # Add text annotations
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                color = 'white' if abs(causality_matrix[i, j]) > 0.5 else 'black'
                plt.text(j, i, f'{causality_matrix[i, j]:.2f}',
                         ha='center', va='center', color=color, fontweight='bold')

    # Highlight expected relationships if provided
    if expected_relationships:
        for (i, j, strength) in expected_relationships:
            plt.gca().add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                              fill=False, edgecolor='green', linewidth=3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_time_series(data_dict, title, save_path=None):
    """Visualize the four time series."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    variables = ['X1', 'X2', 'X3', 'X4']
    colors = ['blue', 'red', 'green', 'purple']

    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i // 2, i % 2]
        ax.plot(data_dict[var][:300], color=color, alpha=0.8)
        ax.set_title(f'{var} Time Series', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Value')
        ax.set_xlabel('Time')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def print_summary(results, simulation_name, expected_relationships):
    """Print a summary of the CCM results."""
    print(f"\n{'=' * 60}")
    print(f"Summary for {simulation_name}")
    print(f"{'=' * 60}")

    variables = ['X1', 'X2', 'X3', 'X4']
    var_to_idx = {var: i for i, var in enumerate(variables)}

    print("\nDetected Relationships (ρ > 0.3):")
    strong_relationships = []

    for var1, var2 in combinations(variables, 2):
        key = f"{var1}_{var2}"
        if key in results and results[key]:
            max_lib = max(results[key]['lib_sizes'])

            # Check both directions
            if max_lib in results[key]['X_predicts_Y']:
                rho = results[key]['X_predicts_Y'][max_lib]['mean']
                if rho > 0.3:
                    print(f"  {var1} → {var2}: ρ = {rho:.3f}")
                    strong_relationships.append((var_to_idx[var1], var_to_idx[var2], rho))

            if max_lib in results[key]['Y_predicts_X']:
                rho = results[key]['Y_predicts_X'][max_lib]['mean']
                if rho > 0.3:
                    print(f"  {var2} → {var1}: ρ = {rho:.3f}")
                    strong_relationships.append((var_to_idx[var2], var_to_idx[var1], rho))

    print(f"\nExpected vs Detected:")
    for (i, j, expected) in expected_relationships:
        found = False
        for (det_i, det_j, det_rho) in strong_relationships:
            if i == det_i and j == det_j:
                status = "✓ DETECTED" if det_rho > 0.3 else "✗ WEAK"
                print(f"  {variables[i]} → {variables[j]}: Expected={expected}, Detected={det_rho:.3f} {status}")
                found = True
                break
        if not found:
            print(f"  {variables[i]} → {variables[j]}: Expected={expected}, Detected=< 0.3 ✗ NOT DETECTED")


def run_all_simulations():
    """Run all three simulations and compare results."""
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = f"causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "simulation_log.txt")

    # Capture ALL output to log file
    with LogCapture(log_file):
        print("Running Three Causality Simulations")
        print("=" * 60)
        print(f"Output will be saved to: {output_dir}")
        print(f"Log file: {log_file}")

        # Simulation 1: Linear Chain
        print("\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
        print("-" * 50)
        data1 = generate_simulation_1(n_points=1500, strength=0.7, delay=3)

        ts_plot1 = os.path.join(output_dir, "sim1_timeseries.png")
        visualize_time_series(data1, "Simulation 1: Linear Chain (1 → 2 → 3 → 4)", save_path=ts_plot1)

        print("  Analyzing causality...")
        results1 = analyze_all_pairs(data1, num_simulations=20)
        expected1 = [(0, 1, "Strong"), (1, 2, "Strong"), (2, 3, "Strong")]  # X1→X2, X2→X3, X3→X4

        causality_plot1 = os.path.join(output_dir, "sim1_causality.png")
        create_causality_plot(results1, "Simulation 1: Linear Chain", expected1, save_path=causality_plot1)
        print_summary(results1, "Simulation 1", expected1)

        # Simulation 2: Hub with Outsider
        print("\n2. Running Simulation 2: Hub + Outsider (1 → 2 strong, 1 → 3 weak, 4 independent)")
        print("-" * 50)
        data2 = generate_simulation_2(n_points=1500, strong_strength=0.8, weak_strength=0.3, delay=3)

        ts_plot2 = os.path.join(output_dir, "sim2_timeseries.png")
        visualize_time_series(data2, "Simulation 2: Hub + Outsider", save_path=ts_plot2)

        print("  Analyzing causality...")
        results2 = analyze_all_pairs(data2, num_simulations=20)
        expected2 = [(0, 1, "Strong"), (0, 2, "Weak")]  # X1→X2 strong, X1→X3 weak

        causality_plot2 = os.path.join(output_dir, "sim2_causality.png")
        create_causality_plot(results2, "Simulation 2: Hub + Outsider", expected2, save_path=causality_plot2)
        print_summary(results2, "Simulation 2", expected2)

        # Simulation 3: Complex Network
        print("\n3. Running Simulation 3: Complex Network (1 → 2 strong, 1 → 3 weak, 4 → 2 weak)")
        print("-" * 50)
        data3 = generate_simulation_3(n_points=1500, strong_strength=0.8, weak_strength=0.3, delay=3)

        ts_plot3 = os.path.join(output_dir, "sim3_timeseries.png")
        visualize_time_series(data3, "Simulation 3: Complex Network", save_path=ts_plot3)

        print("  Analyzing causality...")
        results3 = analyze_all_pairs(data3, num_simulations=20)
        expected3 = [(0, 1, "Strong"), (0, 2, "Weak"), (3, 1, "Weak")]  # X1→X2 strong, X1→X3 weak, X4→X2 weak

        causality_plot3 = os.path.join(output_dir, "sim3_causality.png")
        create_causality_plot(results3, "Simulation 3: Complex Network", expected3, save_path=causality_plot3)
        print_summary(results3, "Simulation 3", expected3)

        # Save results to CSV files
        save_results_to_csv(results1, os.path.join(output_dir, "sim1_results.csv"))
        save_results_to_csv(results2, os.path.join(output_dir, "sim2_results.csv"))
        save_results_to_csv(results3, os.path.join(output_dir, "sim3_results.csv"))

        # Final Comparison
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        print("Simulation 1 (Chain): Sequential causality with equal strength")
        print("Simulation 2 (Hub): One node influences multiple others with different strengths")
        print("Simulation 3 (Network): Multiple nodes contributing to same targets")
        print("\nKey insights:")
        print("- CCM should detect strongest direct relationships most clearly")
        print("- Indirect relationships (X1→X3 in Simulation 1) should be weaker")
        print("- Multiple influences on same target should be detectable but weaker individually")

        print(f"\nAll outputs saved to: {output_dir}")
        print(f"Log file: {log_file}")

    # After logging context is closed, print to console
    print(f"\nSimulation completed! All results saved to: {output_dir}")
    print(f"Check {log_file} for detailed log including any warnings.")


def save_results_to_csv(results, filename):
    """Save CCM results to CSV file."""
    data_rows = []

    for key, result in results.items():
        if result is not None:
            source, target = key.split('_')
            max_lib = max(result['lib_sizes'])

            # Save X predicts Y results
            if max_lib in result['X_predicts_Y']:
                data_rows.append({
                    'Source': source,
                    'Target': target,
                    'Direction': f"{source} → {target}",
                    'Mean_Rho': result['X_predicts_Y'][max_lib]['mean'],
                    'Std_Rho': result['X_predicts_Y'][max_lib]['std'],
                    'CI_Lower': result['X_predicts_Y'][max_lib]['ci_lower'],
                    'CI_Upper': result['X_predicts_Y'][max_lib]['ci_upper'],
                    'Library_Size': max_lib
                })

            # Save Y predicts X results
            if max_lib in result['Y_predicts_X']:
                data_rows.append({
                    'Source': target,
                    'Target': source,
                    'Direction': f"{target} → {source}",
                    'Mean_Rho': result['Y_predicts_X'][max_lib]['mean'],
                    'Std_Rho': result['Y_predicts_X'][max_lib]['std'],
                    'CI_Lower': result['Y_predicts_X'][max_lib]['ci_lower'],
                    'CI_Upper': result['Y_predicts_X'][max_lib]['ci_upper'],
                    'Library_Size': max_lib
                })

    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Saved results to: {filename}")


if __name__ == "__main__":
    run_all_simulations()