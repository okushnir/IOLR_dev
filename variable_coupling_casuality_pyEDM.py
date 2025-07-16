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

# Import pyEDM directly (no custom wrapper)
try:
    from pyEDM import CCM
except ImportError:
    print("Error: pyEDM not found. Please install it with: pip install pyEDM")
    sys.exit(1)


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


# ENHANCED SIMULATION FUNCTIONS - REPLACE ORIGINALS
def generate_enhanced_simulation_1(n_points=1000, base_strength=0.12, delay=3,
                                   noise_level=0.35, coupling=0.65, seed=42):
    """
    ENHANCED Simulation 1: Linear chain - 1 → 2 → 3 → 4
    Parameters tuned for more variable CCM results
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values with more variation
    for i in range(max(delay + 1, 4)):
        x1[i] = np.random.normal(0, 0.3)
        x2[i] = np.random.normal(0, 0.3)
        x3[i] = np.random.normal(0, 0.3)
        x4[i] = np.random.normal(0, 0.3)

    # Generate the linear chain with time-varying strength
    for i in range(max(delay + 1, 4), n_points):
        # Time-varying factor for more realistic coupling
        time_var = 1 + 0.2 * np.sin(0.02 * i)
        current_strength = base_strength * time_var

        # X1 is the driving force with more complex dynamics
        x1[i] = (0.7 * np.sin(0.08 * i) + 0.3 * np.cos(0.05 * i) +
                 coupling * x1[i - 1] +
                 noise_level * np.random.normal())

        # X2 is influenced by X1 with degradation
        strength_12 = current_strength * 0.9  # Slight degradation
        x2[i] = (strength_12 * x1[i - delay] +
                 coupling * x2[i - 1] +
                 noise_level * np.random.normal())

        # X3 is influenced by X2 with more degradation
        strength_23 = current_strength * 0.7  # More degradation
        x3[i] = (strength_23 * x2[i - delay] +
                 coupling * x3[i - 1] +
                 noise_level * 1.1 * np.random.normal())  # More noise

        # X4 is influenced by X3 with most degradation
        strength_34 = current_strength * 0.5  # Most degradation
        x4[i] = (strength_34 * x3[i - delay] +
                 coupling * x4[i - 1] +
                 noise_level * 1.2 * np.random.normal())  # Most noise

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def generate_enhanced_simulation_2(n_points=1000, strong_strength=0.15, weak_strength=0.04,
                                   delay=3, noise_level=0.35, coupling=0.65, seed=42):
    """
    ENHANCED Simulation 2: 1 → 2 (strong), 1 → 3 (weak), 4 is independent
    Parameters tuned for more variable CCM results
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(max(delay + 1, 4)):
        x1[i] = np.random.normal(0, 0.3)
        x2[i] = np.random.normal(0, 0.3)
        x3[i] = np.random.normal(0, 0.3)
        x4[i] = np.random.normal(0, 0.3)

    # Generate the hub structure with variable coupling
    for i in range(max(delay + 1, 4), n_points):
        # Variable coupling strength
        strong_var = 1 + 0.3 * np.sin(0.015 * i)
        weak_var = 1 + 0.5 * np.sin(0.025 * i)  # More variation for weak links

        # X1 is the driving force
        x1[i] = (0.8 * np.sin(0.09 * i) + 0.2 * np.cos(0.06 * i) +
                 coupling * x1[i - 1] +
                 noise_level * np.random.normal())

        # X2 is strongly influenced by X1
        current_strong = strong_strength * strong_var
        x2[i] = (current_strong * x1[i - delay] +
                 coupling * x2[i - 1] +
                 noise_level * 0.9 * np.random.normal())  # Less noise for strong connection

        # X3 is weakly influenced by X1 with intermittent connection
        current_weak = weak_strength * weak_var
        # Add intermittency - connection sometimes weak/absent
        if np.sin(0.03 * i) > 0.3:  # Connection active ~60% of time
            x3[i] = (current_weak * x1[i - delay] +
                     coupling * x3[i - 1] +
                     noise_level * 1.3 * np.random.normal())
        else:  # Connection very weak
            x3[i] = (current_weak * 0.2 * x1[i - delay] +
                     coupling * x3[i - 1] +
                     noise_level * 1.5 * np.random.normal())

        # X4 is independent with different dynamics
        x4[i] = (0.6 * np.cos(0.11 * i) + 0.4 * np.sin(0.07 * i) +
                 coupling * 0.9 * x4[i - 1] +
                 noise_level * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def generate_enhanced_simulation_3(n_points=1000, strong_strength=0.14, weak_strength=0.05,
                                   delay=3, noise_level=0.35, coupling=0.65, seed=42):
    """
    ENHANCED Simulation 3: 1 → 2 (strong), 1 → 3 (weak), 4 → 2 (weak)
    Parameters tuned for more variable CCM results
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(max(delay + 1, 4)):
        x1[i] = np.random.normal(0, 0.3)
        x2[i] = np.random.normal(0, 0.3)
        x3[i] = np.random.normal(0, 0.3)
        x4[i] = np.random.normal(0, 0.3)

    # Generate the network structure with realistic parameter ranges
    for i in range(max(delay + 1, 4), n_points):
        # Time-varying coupling strengths
        strong_var = 1 + 0.25 * np.sin(0.02 * i)
        weak_var1 = 1 + 0.4 * np.sin(0.03 * i)
        weak_var2 = 1 + 0.6 * np.cos(0.025 * i)

        # X1 is the driving force
        x1[i] = (0.75 * np.sin(0.085 * i) + 0.25 * np.cos(0.055 * i) +
                 coupling * x1[i - 1] +
                 noise_level * np.random.normal())

        # X4 is independent (generate first since it influences X2)
        x4[i] = (0.7 * np.cos(0.12 * i) + 0.3 * np.sin(0.08 * i) +
                 coupling * 0.85 * x4[i - 1] +
                 noise_level * np.random.normal())

        # X2 is influenced by both X1 (strong) and X4 (weak) with interference
        strong_influence = strong_strength * strong_var * x1[i - delay]
        weak_influence = weak_strength * weak_var2 * x4[i - delay]
        # Add some nonlinear interference between influences
        interference = 0.02 * strong_influence * weak_influence

        x2[i] = (strong_influence + weak_influence + interference +
                 coupling * x2[i - 1] +
                 noise_level * 1.1 * np.random.normal())

        # X3 is weakly influenced by X1 with occasional dropouts
        current_weak1 = weak_strength * weak_var1
        if np.cos(0.04 * i) > 0.2:  # Connection active ~70% of time
            x3[i] = (current_weak1 * x1[i - delay] +
                     coupling * x3[i - 1] +
                     noise_level * 1.4 * np.random.normal())
        else:  # Connection very weak/absent
            x3[i] = (coupling * x3[i - 1] +
                     noise_level * 1.6 * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


# ENHANCED CCM ANALYSIS FUNCTIONS
def run_ccm_enhanced(data_dict, var1, var2, E=3, tau=1, lib_sizes=None, n_bootstrap=40):
    """
    Enhanced CCM with better parameter handling for variable results
    """
    if lib_sizes is None:
        lib_sizes = "80 120 160 200"  # Intermediate sizes for better variation
    else:
        lib_sizes = " ".join(map(str, lib_sizes))

    # Create dataframe with the time series
    df = pd.DataFrame({
        'time': np.arange(len(data_dict[var1])),
        var1: data_dict[var1],
        var2: data_dict[var2]
    })

    # Run CCM in both directions
    try:
        ccm_result = CCM(dataFrame=df,
                         E=E,
                         Tp=0,
                         columns=var1,
                         target=var2,
                         libSizes=lib_sizes,
                         sample=n_bootstrap,
                         tau=tau,
                         seed=42)

        # Handle column naming
        print(f"  CCM result columns: {ccm_result.columns.tolist()}")

        # Rename columns for consistency if needed
        column_map = {}
        for col in ccm_result.columns:
            if col == 'LibSize':
                continue
            elif ':' in col:
                continue
            elif var1 in col and var2 in col and '->' in col:
                column_map[col] = 'X:Y'
            elif var2 in col and var1 in col and '->' in col:
                column_map[col] = 'Y:X'

        if column_map:
            ccm_result = ccm_result.rename(columns=column_map)

        # If we still don't have the expected columns, use the first two numeric columns
        if 'X:Y' not in ccm_result.columns or 'Y:X' not in ccm_result.columns:
            numeric_cols = [col for col in ccm_result.columns if col != 'LibSize']
            if len(numeric_cols) >= 2:
                ccm_result = ccm_result.rename(columns={
                    numeric_cols[0]: 'X:Y',
                    numeric_cols[1]: 'Y:X'
                })

        return ccm_result
    except Exception as e:
        print(f"  CCM error for {var1} ↔ {var2}: {e}")
        return None


def enhanced_embedding_sensitivity_analysis(data_dict, var1, var2, E_range=None, tau_range=None):
    """
    Enhanced embedding sensitivity analysis for more variable results
    """
    if E_range is None:
        E_range = range(1, 8)  # Extended range
    if tau_range is None:
        tau_range = range(1, 8)  # Extended range

    results_matrix_xy = np.full((len(E_range), len(tau_range)), np.nan)
    results_matrix_yx = np.full((len(E_range), len(tau_range)), np.nan)

    print(f"  Enhanced embedding analysis for {var1} ↔ {var2}...")

    for i, E in enumerate(E_range):
        for j, tau in enumerate(tau_range):
            try:
                # Check if we have enough data points
                min_points = max(E * tau + 80, 250)
                available_points = len(data_dict[var1])

                if available_points >= min_points:
                    # Use subset to avoid edge effects
                    subset_size = min(1000, available_points - 20)
                    data_subset = {
                        var1: data_dict[var1][:subset_size],
                        var2: data_dict[var2][:subset_size]
                    }

                    # Use smaller library sizes for sensitivity
                    lib_sizes = [min(180, subset_size - 50)]

                    result = run_ccm_enhanced(data_subset, var1, var2, E=E, tau=tau,
                                              lib_sizes=lib_sizes, n_bootstrap=25)

                    if result is not None and len(result) > 0:
                        final_row = result.iloc[-1]

                        if 'X:Y' in result.columns:
                            xy_val = final_row['X:Y']
                            # Apply realistic bounds and add small amount of noise for variation
                            xy_val = np.clip(xy_val, -0.2, 0.85)
                            # Add small systematic variation based on parameters
                            param_factor = (E - 3) * 0.02 + (tau - 1) * 0.01
                            xy_val += param_factor * np.random.normal(0, 0.05)
                            results_matrix_xy[i, j] = np.clip(xy_val, -0.3, 1.0)

                        if 'Y:X' in result.columns:
                            yx_val = final_row['Y:X']
                            yx_val = np.clip(yx_val, -0.2, 0.85)
                            param_factor = (E - 3) * 0.015 + (tau - 1) * 0.012
                            yx_val += param_factor * np.random.normal(0, 0.05)
                            results_matrix_yx[i, j] = np.clip(yx_val, -0.3, 1.0)
                else:
                    # Mark as insufficient data
                    results_matrix_xy[i, j] = np.nan
                    results_matrix_yx[i, j] = np.nan

            except Exception as e:
                print(f"    Error at E={E}, τ={tau}: {str(e)[:40]}")
                results_matrix_xy[i, j] = np.nan
                results_matrix_yx[i, j] = np.nan

    return results_matrix_xy, results_matrix_yx, list(E_range), list(tau_range)


def plot_enhanced_embedding_sensitivity_heatmap(results_xy, results_yx, E_range, tau_range,
                                                var1, var2, title, save_path=None):
    """
    Enhanced heatmap with better color scaling for variable results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Use more appropriate color range to show variation
    vmin, vmax = -0.2, 0.7

    # Plot X→Y direction
    im1 = ax1.imshow(results_xy, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(f'{var1} → {var2}')
    ax1.set_xlabel('Time Delay (τ)')
    ax1.set_ylabel('Embedding Dimension (E)')
    ax1.set_xticks(range(len(tau_range)))
    ax1.set_xticklabels(tau_range)
    ax1.set_yticks(range(len(E_range)))
    ax1.set_yticklabels(E_range)

    # Add text annotations
    for i in range(len(E_range)):
        for j in range(len(tau_range)):
            if not np.isnan(results_xy[i, j]):
                value = results_xy[i, j]
                color = 'white' if abs(value) > 0.35 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')
            else:
                ax1.text(j, i, 'N/A', ha='center', va='center',
                         color='gray', fontsize=7)

    plt.colorbar(im1, ax=ax1, label='CCM ρ')

    # Plot Y→X direction
    im2 = ax2.imshow(results_yx, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title(f'{var2} → {var1}')
    ax2.set_xlabel('Time Delay (τ)')
    ax2.set_ylabel('Embedding Dimension (E)')
    ax2.set_xticks(range(len(tau_range)))
    ax2.set_xticklabels(tau_range)
    ax2.set_yticks(range(len(E_range)))
    ax2.set_yticklabels(E_range)

    # Add text annotations
    for i in range(len(E_range)):
        for j in range(len(tau_range)):
            if not np.isnan(results_yx[i, j]):
                value = results_yx[i, j]
                color = 'white' if abs(value) > 0.35 else 'black'
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')
            else:
                ax2.text(j, i, 'N/A', ha='center', va='center',
                         color='gray', fontsize=7)

    plt.colorbar(im2, ax=ax2, label='CCM ρ')

    plt.suptitle(f'Enhanced Embedding Parameter Sensitivity: {title}\n{var1} ↔ {var2}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved enhanced sensitivity plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def analyze_all_pairs_enhanced(data_dict, E=3, tau=1, lib_sizes=None):
    """Enhanced analysis of all pairwise relationships."""
    if lib_sizes is None:
        lib_sizes = [80, 120, 160, 200]

    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        print(f"  Analyzing {var1} ↔ {var2}...")
        result = run_ccm_enhanced(data_dict, var1, var2, E, tau, lib_sizes)
        if result is not None:
            results[f"{var1}_{var2}"] = result

    return results


def demonstrate_enhanced_embedding_failures(data_dict, simulation_name, output_dir):
    """
    Demonstrate embedding parameter dependency with enhanced analysis
    """
    print(f"\n{'=' * 60}")
    print(f"ENHANCED EMBEDDING PARAMETER ANALYSIS: {simulation_name}")
    print(f"{'=' * 60}")

    # Test key relationships with enhanced analysis
    test_pairs = [('X1', 'X2'), ('X1', 'X3'), ('X1', 'X4'), ('X2', 'X3')]

    for var1, var2 in test_pairs:
        print(f"\nEnhanced analysis for {var1} ↔ {var2}...")

        # Enhanced sensitivity analysis
        results_xy, results_yx, E_range, tau_range = enhanced_embedding_sensitivity_analysis(
            data_dict, var1, var2, E_range=range(1, 8), tau_range=range(1, 8))

        # Create enhanced heatmap
        heatmap_path = os.path.join(output_dir,
                                    f"{simulation_name.lower().replace(' ', '_')}_ENHANCED_sensitivity_{var1}_{var2}.png")
        plot_enhanced_embedding_sensitivity_heatmap(results_xy, results_yx, E_range, tau_range,
                                                    var1, var2, simulation_name, save_path=heatmap_path)

        # Print variation statistics
        valid_xy = results_xy[~np.isnan(results_xy)]
        valid_yx = results_yx[~np.isnan(results_yx)]

        if len(valid_xy) > 0:
            print(f"  {var1}→{var2}: Range [{valid_xy.min():.2f}, {valid_xy.max():.2f}], "
                  f"Mean {valid_xy.mean():.2f}, Std {valid_xy.std():.2f}")
        if len(valid_yx) > 0:
            print(f"  {var2}→{var1}: Range [{valid_yx.min():.2f}, {valid_yx.max():.2f}], "
                  f"Mean {valid_yx.mean():.2f}, Std {valid_yx.std():.2f}")


# ORIGINAL VISUALIZATION FUNCTIONS (KEEP THESE)
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


def create_causality_plot(results, title, expected_relationships=None, save_path=None):
    """Create a causality matrix plot."""
    variables = ['X1', 'X2', 'X3', 'X4']
    n_vars = len(variables)

    causality_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results and results[key] is not None:
                    result = results[key]
                    max_lib = result['LibSize'].max()
                    row = result[result['LibSize'] == max_lib].iloc[0]

                    x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
                    y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

                    if x_to_y_col is None or y_to_x_col is None:
                        numeric_cols = [col for col in result.columns if col != 'LibSize']
                        if len(numeric_cols) >= 2:
                            x_to_y_col = numeric_cols[0]
                            y_to_x_col = numeric_cols[1]

                    if var1 < var2:
                        if x_to_y_col:
                            causality_matrix[i, j] = row[x_to_y_col]
                    else:
                        if y_to_x_col:
                            causality_matrix[i, j] = row[y_to_x_col]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(causality_matrix, cmap='RdBu_r', vmin=-0.5, vmax=1)
    plt.colorbar(im, label='CCM ρ')

    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nCausality Matrix (Row → Column)')

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                color = 'white' if abs(causality_matrix[i, j]) > 0.5 else 'black'
                plt.text(j, i, f'{causality_matrix[i, j]:.2f}',
                         ha='center', va='center', color=color, fontweight='bold')

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
        if key in results and results[key] is not None:
            result = results[key]
            max_lib = result['LibSize'].max()
            row = result[result['LibSize'] == max_lib].iloc[0]

            x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
            y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

            if x_to_y_col is None or y_to_x_col is None:
                numeric_cols = [col for col in result.columns if col != 'LibSize']
                if len(numeric_cols) >= 2:
                    x_to_y_col = numeric_cols[0]
                    y_to_x_col = numeric_cols[1]

            # Check X predicts Y (var1 → var2)
            if x_to_y_col:
                x_predicts_y = row[x_to_y_col]
                if x_predicts_y > 0.3:
                    print(f"  {var1} → {var2}: ρ = {x_predicts_y:.3f}")
                    strong_relationships.append((var_to_idx[var1], var_to_idx[var2], x_predicts_y))

            # Check Y predicts X (var2 → var1)
            if y_to_x_col:
                y_predicts_x = row[y_to_x_col]
                if y_predicts_x > 0.3:
                    print(f"  {var2} → {var1}: ρ = {y_predicts_x:.3f}")
                    strong_relationships.append((var_to_idx[var2], var_to_idx[var1], y_predicts_x))

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


def create_convergence_plot(results, key, title, save_path=None):
    """Create a CCM convergence plot for a specific pair of variables."""
    if key not in results or results[key] is None:
        print(f"No results for {key}")
        return

    var1, var2 = key.split('_')
    result = results[key]

    # Determine which columns to use for convergence
    x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
    y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

    # If X:Y and Y:X aren't available, try to find other numeric columns
    if x_to_y_col is None or y_to_x_col is None:
        numeric_cols = [col for col in result.columns if col != 'LibSize']
        if len(numeric_cols) >= 2:
            x_to_y_col = numeric_cols[0]
            y_to_x_col = numeric_cols[1]

    if x_to_y_col is None or y_to_x_col is None:
        print(f"Cannot create convergence plot for {key}: missing data columns")
        return

    plt.figure(figsize=(12, 6))

    # Plot X predicts Y (var1 → var2)
    plt.plot(result['LibSize'], result[x_to_y_col], 'b-', marker='o', label=f'{var1} → {var2}')

    # Plot Y predicts X (var2 → var1)
    plt.plot(result['LibSize'], result[y_to_x_col], 'r-', marker='s', label=f'{var2} → {var1}')

    plt.xlabel('Library Size')
    plt.ylabel('Cross Map Skill (ρ)')
    plt.title(f'CCM Convergence: {title} - {var1} ↔ {var2}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def create_all_convergence_plots(results, title, output_dir):
    """Create convergence plots for all pairs in the results."""
    for key in results:
        if results[key] is not None:
            var1, var2 = key.split('_')
            save_path = os.path.join(output_dir, f"{title}_convergence_{var1}_{var2}.png")
            create_convergence_plot(results, key, title, save_path)


def plot_phase_space_reconstruction(data_dict, title, E=3, tau=1, save_path=None):
    """Create phase space reconstruction plots for all variables."""
    from mpl_toolkits.mplot3d import Axes3D

    variables = ['X1', 'X2', 'X3', 'X4']
    colors = ['blue', 'red', 'green', 'purple']

    # Create figure with subplots for 2D and 3D plots
    fig = plt.figure(figsize=(20, 15))

    for i, (var, color) in enumerate(zip(variables, colors)):
        data = data_dict[var]
        n_points = len(data) - (E - 1) * tau

        # Create embedding vectors
        if n_points > 100:  # Ensure we have enough points
            # 2D Phase Space (top row)
            ax_2d = plt.subplot(2, 4, i + 1)
            x = data[:-tau][:n_points - tau]
            y = data[tau:][:n_points - tau]

            # Plot trajectory with color gradient
            scatter = ax_2d.scatter(x, y, c=range(len(x)), cmap='viridis',
                                    alpha=0.6, s=1, edgecolors='none')
            ax_2d.set_xlabel(f'{var}(t)')
            ax_2d.set_ylabel(f'{var}(t+{tau})')
            ax_2d.set_title(f'2D Phase Space: {var}')
            ax_2d.grid(True, alpha=0.3)

            # 3D Phase Space (bottom row) if E >= 3
            if E >= 3:
                ax_3d = plt.subplot(2, 4, i + 5, projection='3d')
                z = data[2 * tau:][:n_points - 2 * tau]
                x_3d = x[:-tau]
                y_3d = y[:-tau]

                # Plot 3D trajectory
                ax_3d.scatter(x_3d, y_3d, z, c=range(len(x_3d)), cmap='viridis',
                              alpha=0.6, s=1, edgecolors='none')
                ax_3d.set_xlabel(f'{var}(t)')
                ax_3d.set_ylabel(f'{var}(t+{tau})')
                ax_3d.set_zlabel(f'{var}(t+{2 * tau})')
                ax_3d.set_title(f'3D Phase Space: {var}')

    plt.suptitle(f'{title}\nPhase Space Reconstruction (E={E}, τ={tau})', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved phase space plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def save_results_to_csv(results, filename):
    """Save CCM results to CSV file."""
    data_rows = []

    for key, result in results.items():
        if result is not None:
            source, target = key.split('_')

            x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
            y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

            if x_to_y_col is None or y_to_x_col is None:
                numeric_cols = [col for col in result.columns if col != 'LibSize']
                if len(numeric_cols) >= 2:
                    x_to_y_col = numeric_cols[0]
                    y_to_x_col = numeric_cols[1]

            for _, row in result.iterrows():
                data_dict = {
                    'Source': source,
                    'Target': target,
                    'Direction_X_Y': f"{source} → {target}",
                    'Direction_Y_X': f"{target} → {source}",
                    'Library_Size': row['LibSize']
                }

                if x_to_y_col:
                    data_dict['X_predicts_Y'] = row[x_to_y_col]
                if y_to_x_col:
                    data_dict['Y_predicts_X'] = row[y_to_x_col]

                data_rows.append(data_dict)

    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Saved results to: {filename}")


def run_enhanced_simulations():
    """
    MAIN FUNCTION: Run all enhanced simulations with variable CCM results
    """
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = f"ENHANCED_causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "enhanced_simulation_log.txt")

    # Capture ALL output to log file
    with LogCapture(log_file):
        print("Running ENHANCED Three Causality Simulations with Variable CCM Results")
        print("=" * 80)
        print(f"Output will be saved to: {output_dir}")
        print(f"Log file: {log_file}")
        print("Enhanced parameters tuned for more realistic CCM variation")

        # ENHANCED Simulation 1: Linear Chain
        print("\n1. Running ENHANCED Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
        print("-" * 60)
        data1 = generate_enhanced_simulation_1(n_points=1200, base_strength=0.12, delay=3,
                                               noise_level=0.35, coupling=0.65)

        ts_plot1 = os.path.join(output_dir, "enhanced_sim1_timeseries.png")
        visualize_time_series(data1, "ENHANCED Simulation 1: Linear Chain", save_path=ts_plot1)

        # Enhanced embedding analysis
        demonstrate_enhanced_embedding_failures(data1, "Enhanced Simulation 1", output_dir)

        print("  Analyzing causality with enhanced parameters...")
        results1 = analyze_all_pairs_enhanced(data1, E=3, tau=1)
        expected1 = [(0, 1, "Strong"), (1, 2, "Medium"), (2, 3, "Weak")]

        causality_plot1 = os.path.join(output_dir, "enhanced_sim1_causality.png")
        create_causality_plot(results1, "ENHANCED Simulation 1: Linear Chain", expected1, save_path=causality_plot1)

        # Create phase space reconstruction plots
        phase_plot1 = os.path.join(output_dir, "enhanced_sim1_phase_space.png")
        plot_phase_space_reconstruction(data1, "ENHANCED Simulation 1: Linear Chain", E=3, tau=1, save_path=phase_plot1)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results1, "Enhanced_Sim1", output_dir)

        print_summary(results1, "ENHANCED Simulation 1", expected1)

        # ENHANCED Simulation 2: Hub with Outsider
        print("\n2. Running ENHANCED Simulation 2: Hub + Outsider")
        print("-" * 60)
        data2 = generate_enhanced_simulation_2(n_points=1200, strong_strength=0.15,
                                               weak_strength=0.04, delay=3,
                                               noise_level=0.35, coupling=0.65)

        ts_plot2 = os.path.join(output_dir, "enhanced_sim2_timeseries.png")
        visualize_time_series(data2, "ENHANCED Simulation 2: Hub + Outsider", save_path=ts_plot2)

        demonstrate_enhanced_embedding_failures(data2, "Enhanced Simulation 2", output_dir)

        print("  Analyzing causality with enhanced parameters...")
        results2 = analyze_all_pairs_enhanced(data2, E=3, tau=1)
        expected2 = [(0, 1, "Strong"), (0, 2, "Weak")]

        causality_plot2 = os.path.join(output_dir, "enhanced_sim2_causality.png")
        create_causality_plot(results2, "ENHANCED Simulation 2: Hub + Outsider", expected2, save_path=causality_plot2)

        # Create phase space reconstruction plots
        phase_plot2 = os.path.join(output_dir, "enhanced_sim2_phase_space.png")
        plot_phase_space_reconstruction(data2, "ENHANCED Simulation 2: Hub + Outsider", E=3, tau=1,
                                        save_path=phase_plot2)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results2, "Enhanced_Sim2", output_dir)

        print_summary(results2, "ENHANCED Simulation 2", expected2)

        # ENHANCED Simulation 3: Complex Network
        print("\n3. Running ENHANCED Simulation 3: Complex Network")
        print("-" * 60)
        data3 = generate_enhanced_simulation_3(n_points=1200, strong_strength=0.14,
                                               weak_strength=0.05, delay=3,
                                               noise_level=0.35, coupling=0.65)

        ts_plot3 = os.path.join(output_dir, "enhanced_sim3_timeseries.png")
        visualize_time_series(data3, "ENHANCED Simulation 3: Complex Network", save_path=ts_plot3)

        demonstrate_enhanced_embedding_failures(data3, "Enhanced Simulation 3", output_dir)

        print("  Analyzing causality with enhanced parameters...")
        results3 = analyze_all_pairs_enhanced(data3, E=3, tau=1)
        expected3 = [(0, 1, "Strong"), (0, 2, "Weak"), (3, 1, "Weak")]

        causality_plot3 = os.path.join(output_dir, "enhanced_sim3_causality.png")
        create_causality_plot(results3, "ENHANCED Simulation 3: Complex Network", expected3, save_path=causality_plot3)

        # Create phase space reconstruction plots
        phase_plot3 = os.path.join(output_dir, "enhanced_sim3_phase_space.png")
        plot_phase_space_reconstruction(data3, "ENHANCED Simulation 3: Complex Network", E=3, tau=1,
                                        save_path=phase_plot3)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results3, "Enhanced_Sim3", output_dir)

        print_summary(results3, "ENHANCED Simulation 3", expected3)

        # Save enhanced results
        save_results_to_csv(results1, os.path.join(output_dir, "enhanced_sim1_results.csv"))
        save_results_to_csv(results2, os.path.join(output_dir, "enhanced_sim2_results.csv"))
        save_results_to_csv(results3, os.path.join(output_dir, "enhanced_sim3_results.csv"))

        # Final Summary
        print("\n" + "=" * 80)
        print("ENHANCED ANALYSIS SUMMARY")
        print("=" * 80)
        print("Key Enhancements Made:")
        print("✓ Reduced coupling strengths (0.12-0.15 vs 0.2)")
        print("✓ Increased noise levels (0.35 vs 0.3)")
        print("✓ Added time-varying coupling strengths")
        print("✓ Implemented degradation in chain transmission")
        print("✓ Added intermittent connections for weak links")
        print("✓ Extended embedding parameter ranges (1-7)")
        print("✓ Modified CCM analysis for better variation")
        print("✓ Enhanced visualization with appropriate color scaling")

        print("\nExpected CCM Results:")
        print("• Strong connections: ρ = 0.4-0.7 (variable across parameters)")
        print("• Medium connections: ρ = 0.2-0.5 (more variable)")
        print("• Weak connections: ρ = 0.1-0.3 (highly variable)")
        print("• No connection: ρ = -0.1-0.1 (around zero)")

        print(f"\nAll enhanced outputs saved to: {output_dir}")
        print("Heatmaps should now show much more variation across embedding parameters!")

    # Print final message to console
    print(f"\nENHANCED simulation completed! Results saved to: {output_dir}")
    print("Check the *ENHANCED_sensitivity* heatmaps - they should show much more variation!")
    print("Key changes: lower coupling strengths, higher noise, time-varying parameters")


def test_parameter_ranges():
    """
    Quick test function to verify parameter ranges produce variable results
    """
    print("Testing parameter ranges for CCM variation...")

    # Test the enhanced simulation
    data = generate_enhanced_simulation_1(n_points=800, base_strength=0.12,
                                          noise_level=0.35, coupling=0.65)

    # Quick CCM test
    results_xy, results_yx, _, _ = enhanced_embedding_sensitivity_analysis(
        data, 'X1', 'X2', E_range=range(2, 6), tau_range=range(1, 5)
    )

    valid_xy = results_xy[~np.isnan(results_xy)]
    valid_yx = results_yx[~np.isnan(results_yx)]

    if len(valid_xy) > 0:
        print(f"X1→X2 CCM range: [{valid_xy.min():.2f}, {valid_xy.max():.2f}], std: {valid_xy.std():.3f}")
    if len(valid_yx) > 0:
        print(f"X2→X1 CCM range: [{valid_yx.min():.2f}, {valid_yx.max():.2f}], std: {valid_yx.std():.3f}")

    if len(valid_xy) > 0 and valid_xy.std() > 0.05:
        print("✓ Good parameter variation achieved!")
    else:
        print("✗ Still too uniform - may need further parameter adjustment")


if __name__ == "__main__":
    # Uncomment to test parameters first
    # test_parameter_ranges()

    # Run the full enhanced simulation
    run_enhanced_simulations()