import warnings

warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import logging

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


def generate_causal_chain(n_points=1000, strength=0.5, delay=3, noise_level=0.1, coupling=0.7, seed=42):
    """
    Generate a causal chain: X1 → X2 → X3 → X4

    Parameters:
    -----------
    n_points : int
        Length of time series
    strength : float
        Strength of causal influence between consecutive variables (0-1)
    delay : int
        Time delay for causal effect
    noise_level : float
        Amount of noise to add
    coupling : float
        Autocorrelation strength for each variable
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Dictionary with keys 'X1', 'X2', 'X3', 'X4' containing time series
    """

    np.random.seed(seed)

    # Initialize arrays
    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    x1[0] = np.random.random()
    x2[0] = np.random.random()
    x3[0] = np.random.random()
    x4[0] = np.random.random()

    # Generate the causal chain
    for i in range(1, n_points):
        # X1 is the driving force (independent)
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X2 is influenced by X1
        if i > delay:
            x2[i] = strength * x1[i - delay] + coupling * x2[i - 1] + noise_level * np.random.random()
        else:
            x2[i] = coupling * x2[i - 1] + noise_level * np.random.random()

        # X3 is influenced by X2
        if i > delay * 2:  # X3 receives X1's influence with delay*2
            x3[i] = strength * x2[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()
        else:
            x3[i] = coupling * x3[i - 1] + noise_level * np.random.random()

        # X4 is influenced by X3
        if i > delay * 3:  # X4 receives X1's influence with delay*3
            x4[i] = strength * x3[i - delay] + coupling * x4[i - 1] + noise_level * np.random.random()
        else:
            x4[i] = coupling * x4[i - 1] + noise_level * np.random.random()

    return {
        'X1': x1,
        'X2': x2,
        'X3': x3,
        'X4': x4
    }


def analyze_causal_chain(data_dict, num_simulations=20, lib_sizes=None):
    """
    Analyze all pairwise relationships in the causal chain using CCM.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing time series for X1, X2, X3, X4
    num_simulations : int
        Number of CCM simulations to run
    lib_sizes : list
        Library sizes to test

    Returns:
    --------
    dict : Results for all pairwise comparisons
    """

    # Import the CCM function
    from pyedm_ccm_fixed import run_ccm_simulations

    if lib_sizes is None:
        lib_sizes = [50, 100, 150, 200, 250]

    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    # Analyze all pairwise relationships
    for var1, var2 in combinations(variables, 2):
        print(f"\nAnalyzing {var1} ↔ {var2}...")

        result = run_ccm_simulations(
            time_series_1=data_dict[var1],
            time_series_2=data_dict[var2],
            num_simulations=num_simulations,
            lib_sizes=lib_sizes,
            verbose=False
        )

        results[f"{var1}_{var2}"] = result

    return results


def visualize_causal_chain(data_dict, results=None):
    """
    Visualize the causal chain data and results.
    """

    # Plot the time series
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    variables = ['X1', 'X2', 'X3', 'X4']
    colors = ['blue', 'red', 'green', 'purple']

    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i // 2, i % 2]
        ax.plot(data_dict[var][:200], color=color, alpha=0.8)
        ax.set_title(f'{var} Time Series')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot CCM results if provided
    if results:
        plot_ccm_matrix(results)


def plot_ccm_matrix(results):
    """
    Create a matrix plot showing CCM results for all pairs.
    """
    variables = ['X1', 'X2', 'X3', 'X4']
    n_vars = len(variables)

    # Create matrices for X→Y and Y→X causality
    causality_xy = np.zeros((n_vars, n_vars))
    causality_yx = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results:
                    result = results[key]
                    max_lib = max(result['lib_sizes'])

                    if var1 < var2:  # Original order
                        if max_lib in result['X_predicts_Y']:
                            causality_xy[i, j] = result['X_predicts_Y'][max_lib]['mean']
                        if max_lib in result['Y_predicts_X']:
                            causality_yx[i, j] = result['Y_predicts_X'][max_lib]['mean']
                    else:  # Swapped order
                        if max_lib in result['Y_predicts_X']:
                            causality_xy[i, j] = result['Y_predicts_X'][max_lib]['mean']
                        if max_lib in result['X_predicts_Y']:
                            causality_yx[i, j] = result['X_predicts_Y'][max_lib]['mean']

    # Plot the causality matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # X causes Y matrix
    im1 = ax1.imshow(causality_xy, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(n_vars))
    ax1.set_yticks(range(n_vars))
    ax1.set_xticklabels(variables)
    ax1.set_yticklabels(variables)
    ax1.set_title('Causality Matrix (Row → Column)')
    ax1.set_xlabel('Target Variable')
    ax1.set_ylabel('Source Variable')

    # Add text annotations
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                ax1.text(j, i, f'{causality_xy[i, j]:.2f}',
                         ha='center', va='center', color='black')

    plt.colorbar(im1, ax=ax1, label='CCM ρ')

    # Create a summary plot showing expected vs detected causality
    ax2.bar(range(3), [causality_xy[0, 1], causality_xy[1, 2], causality_xy[2, 3]],
            alpha=0.7, label='Direct causality')
    ax2.bar(range(3), [causality_xy[0, 2], causality_xy[1, 3], causality_xy[0, 3]],
            alpha=0.7, label='Indirect causality')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['X1→X2', 'X2→X3', 'X3→X4'])
    ax2.set_ylabel('CCM ρ')
    ax2.set_title('Direct vs Indirect Causality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_causal_chain_detection():
    """
    Test whether CCM can detect both direct and indirect causality in a chain.
    """
    print("Testing causal chain detection: X1 → X2 → X3 → X4")
    print("=" * 60)

    # Generate causal chain data
    data = generate_causal_chain(
        n_points=1000,
        strength=0.7,  # Strong direct causality
        delay=3,  # 3-step delay
        noise_level=0.1,  # Moderate noise
        coupling=0.8  # Strong autocorrelation
    )

    # Visualize the generated data
    visualize_causal_chain(data)

    # Analyze all pairwise relationships
    results = analyze_causal_chain(data, num_simulations=15)

    # Print summary of results
    print("\nCCM Results Summary:")
    print("=" * 60)

    expected_direct = [('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4')]
    expected_indirect = [('X1', 'X3'), ('X1', 'X4'), ('X2', 'X4')]

    print("\nDirect Relationships (Expected):")
    for var1, var2 in expected_direct:
        key = f"{var1}_{var2}"
        if key in results and results[key]:
            max_lib = max(results[key]['lib_sizes'])
            if max_lib in results[key]['X_predicts_Y']:
                rho = results[key]['X_predicts_Y'][max_lib]['mean']
                print(f"  {var1} → {var2}: ρ = {rho:.3f}")

    print("\nIndirect Relationships (Expected but weaker):")
    for var1, var2 in expected_indirect:
        key = f"{var1}_{var2}"
        if key in results and results[key]:
            max_lib = max(results[key]['lib_sizes'])
            if max_lib in results[key]['X_predicts_Y']:
                rho = results[key]['X_predicts_Y'][max_lib]['mean']
                print(f"  {var1} → {var2}: ρ = {rho:.3f}")

    # Visualize all results
    visualize_causal_chain(data, results)

    return data, results


# Example usage
if __name__ == "__main__":
    # Test the causal chain detection
    data, results = test_causal_chain_detection()

    # You can also create your own custom chain
    print("\n" + "=" * 60)
    print("Creating custom causal chain...")

    custom_data = generate_causal_chain(
        n_points=1500,
        strength=0.8,  # Very strong causality
        delay=5,  # Longer delay
        noise_level=0.05  # Less noise
    )

    # Analyze specific pairs
    from pyedm_ccm_fixed import run_ccm_simulations, plot_ccm_results

    # Test X1 → X4 (indirect causality through X2 and X3)
    print("\nTesting X1 → X4 (indirect causality)...")
    x1_x4_results = run_ccm_simulations(
        custom_data['X1'],
        custom_data['X4'],
        num_simulations=20,
        lib_sizes=[100, 200, 300, 400],
        verbose=True
    )

    if x1_x4_results:
        plot_ccm_results(x1_x4_results, "X1 → X4 (Indirect Causality)")