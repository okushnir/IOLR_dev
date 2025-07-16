import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import warnings
import logging
import sys
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')
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


def create_sequences(data, sequence_length):
    """Create sequences for neural network training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def run_granger_causality_test(data_dict, var1, var2, max_lags=5):
    """
    Granger causality test using linear regression
    More reliable than MLP for causality detection
    """
    from sklearn.linear_model import LinearRegression

    series1 = np.array(data_dict[var1])
    series2 = np.array(data_dict[var2])

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    data_combined = np.column_stack([series1, series2])
    data_scaled = scaler.fit_transform(data_combined)
    series1_scaled = data_scaled[:, 0]
    series2_scaled = data_combined[:, 1]

    best_score = 0
    best_lag = 1

    # Test different lag lengths
    for lag in range(1, max_lags + 1):
        if len(series1_scaled) <= lag + 50:
            continue

        # Create lagged features
        X_restricted = []  # Only past values of Y
        X_full = []  # Past values of both X and Y
        y = []

        for i in range(lag, len(series2_scaled) - 10):
            # Restricted model: only past Y values
            x_restricted = [series2_scaled[i - j] for j in range(1, lag + 1)]
            X_restricted.append(x_restricted)

            # Full model: past Y and X values
            x_full = []
            for j in range(1, lag + 1):
                x_full.extend([series2_scaled[i - j], series1_scaled[i - j]])
            X_full.append(x_full)

            y.append(series2_scaled[i])

        X_restricted = np.array(X_restricted)
        X_full = np.array(X_full)
        y = np.array(y)

        if len(y) < 50:
            continue

        # Split data
        split = int(0.7 * len(y))

        # Restricted model (Y only)
        model_restricted = LinearRegression()
        model_restricted.fit(X_restricted[:split], y[:split])
        pred_restricted = model_restricted.predict(X_restricted[split:])
        r2_restricted = r2_score(y[split:], pred_restricted)

        # Full model (Y + X)
        model_full = LinearRegression()
        model_full.fit(X_full[:split], y[:split])
        pred_full = model_full.predict(X_full[split:])
        r2_full = r2_score(y[split:], pred_full)

        # Granger causality strength
        causality_strength = max(0, r2_full - r2_restricted)

        if causality_strength > best_score:
            best_score = causality_strength
            best_lag = lag

    # Scale the final score
    causality_score = min(1.0, best_score * 20)  # Aggressive scaling

    return {
        'cross_r2': r2_full if 'r2_full' in locals() else 0,
        'self_r2': r2_restricted if 'r2_restricted' in locals() else 0,
        'causality_strength': best_score,
        'causality_score': causality_score,
        'best_lag': best_lag
    }


def analyze_all_pairs_granger(data_dict):
    """Granger causality analysis for all pairs"""
    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        print(f"  Granger test {var1} ↔ {var2}...")

        result_xy = run_granger_causality_test(data_dict, var1, var2)
        result_yx = run_granger_causality_test(data_dict, var2, var1)

        results[f"{var1}_{var2}"] = {
            f"{var1}_to_{var2}": result_xy,
            f"{var2}_to_{var1}": result_yx
        }

    return results


def run_simple_nn_simulations():
    """Enhanced main function with Granger causality"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"GRANGER_causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Running Granger Causality Analysis")
    print("=" * 50)
    print(f"Output: {output_dir}")

    # Simulation 1: Linear Chain
    print("\n1. Linear Chain (1 → 2 → 3 → 4)")
    data1 = generate_enhanced_simulation_1(n_points=1200)

    ts_plot1 = os.path.join(output_dir, "granger_sim1_timeseries.png")
    visualize_time_series(data1, "Granger Sim 1: Linear Chain", save_path=ts_plot1)

    results1 = analyze_all_pairs_granger(data1)

    causality_plot1 = os.path.join(output_dir, "granger_sim1_causality.png")
    create_granger_causality_plot(results1, "Granger Sim 1: Linear Chain", save_path=causality_plot1)

    create_all_convergence_plots(results1, "Granger Sim1", output_dir)
    save_results_to_csv(results1, os.path.join(output_dir, "granger_sim1_results.csv"))

    print_granger_summary(results1, "Linear Chain")

    # Simulation 2: Hub + Outsider
    print("\n2. Hub + Outsider")
    data2 = generate_enhanced_simulation_2(n_points=1200)

    ts_plot2 = os.path.join(output_dir, "granger_sim2_timeseries.png")
    visualize_time_series(data2, "Granger Sim 2: Hub + Outsider", save_path=ts_plot2)

    results2 = analyze_all_pairs_granger(data2)

    causality_plot2 = os.path.join(output_dir, "granger_sim2_causality.png")
    create_granger_causality_plot(results2, "Granger Sim 2: Hub + Outsider", save_path=causality_plot2)

    create_all_convergence_plots(results2, "Granger Sim2", output_dir)
    save_results_to_csv(results2, os.path.join(output_dir, "granger_sim2_results.csv"))

    print_granger_summary(results2, "Hub + Outsider")

    print(f"\nCompleted! Check {output_dir}")


def create_granger_causality_plot(results, title, save_path=None):
    """Causality matrix for Granger results"""
    variables = ['X1', 'X2', 'X3', 'X4']
    n_vars = len(variables)

    causality_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results:
                    result = results[key]
                    if var1 < var2:
                        causality_matrix[i, j] = result[f"{var1}_to_{var2}"]['causality_score']
                        causality_matrix[j, i] = result[f"{var2}_to_{var1}"]['causality_score']

    plt.figure(figsize=(10, 8))
    im = plt.imshow(causality_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(im, label='Granger Causality Score')

    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nGranger Causality Matrix (Row → Column)')

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                color = 'white' if causality_matrix[i, j] > 0.5 else 'black'
                plt.text(j, i, f'{causality_matrix[i, j]:.2f}',
                         ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def print_granger_summary(results, simulation_name):
    """Print Granger results summary"""
    print(f"\nGranger Summary - {simulation_name}")
    print("-" * 40)

    for var1, var2 in combinations(['X1', 'X2', 'X3', 'X4'], 2):
        key = f"{var1}_{var2}"
        if key in results:
            result = results[key]

            score_xy = result[f"{var1}_to_{var2}"]['causality_score']
            score_yx = result[f"{var2}_to_{var1}"]['causality_score']

            if score_xy > 0.3:
                print(f"{var1} → {var2}: {score_xy:.3f}")
            if score_yx > 0.3:
                print(f"{var2} → {var1}: {score_yx:.3f}")


# Add missing visualization functions
def create_convergence_plot(results, var1, var2, title, save_path=None):
    """Create convergence-like plot showing R² comparison"""
    key = f"{var1}_{var2}"
    if key not in results:
        return

    result = results[key]
    xy_result = result[f"{var1}_to_{var2}"]
    yx_result = result[f"{var2}_to_{var1}"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: R² comparison
    metrics = ['Cross R²', 'Self R²', 'Causality Score']
    xy_values = [xy_result['cross_r2'], xy_result['self_r2'], xy_result['causality_score']]
    yx_values = [yx_result['cross_r2'], yx_result['self_r2'], yx_result['causality_score']]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width / 2, xy_values, width, label=f'{var1} → {var2}', alpha=0.8)
    ax1.bar(x + width / 2, yx_values, width, label=f'{var2} → {var1}', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title(f'MLP Performance: {var1} ↔ {var2}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Causality strength comparison
    directions = [f'{var1}→{var2}', f'{var2}→{var1}']
    strengths = [xy_result['causality_strength'], yx_result['causality_strength']]

    bars = ax2.bar(directions, strengths, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Causality Strength')
    ax2.set_title('Causality Strength Comparison')
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, strengths):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.suptitle(f'{title}: {var1} ↔ {var2}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to: {save_path}")
    plt.close()


def create_all_convergence_plots(results, title, output_dir):
    """Create convergence plots for all pairs"""
    variables = ['X1', 'X2', 'X3', 'X4']
    for var1, var2 in combinations(variables, 2):
        save_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_convergence_{var1}_{var2}.png")
        create_convergence_plot(results, var1, var2, title, save_path)


def save_results_to_csv(results, filename):
    """Save results to CSV"""
    data_rows = []
    for key, result in results.items():
        var1, var2 = key.split('_')
        xy_result = result[f"{var1}_to_{var2}"]
        yx_result = result[f"{var2}_to_{var1}"]

        data_rows.append({
            'Source': var1, 'Target': var2,
            'Cross_R2': xy_result['cross_r2'],
            'Self_R2': xy_result['self_r2'],
            'Causality_Score': xy_result['causality_score']
        })

        data_rows.append({
            'Source': var2, 'Target': var1,
            'Cross_R2': yx_result['cross_r2'],
            'Self_R2': yx_result['self_r2'],
            'Causality_Score': yx_result['causality_score']
        })

    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Saved results to: {filename}")


def run_simple_nn_simulations():
    """Enhanced main function with all plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"SIMPLE_NN_causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Running Enhanced Causality Simulations with MLP Neural Networks")
    print("=" * 70)
    print(f"Output saved to: {output_dir}")

    # Simulation 1: Linear Chain
    print("\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
    print("-" * 50)
    data1 = generate_enhanced_simulation_1(n_points=1200)

    # Debug key relationships
    test_individual_relationship(data1, 'X1', 'X2')
    test_individual_relationship(data1, 'X2', 'X3')
    test_individual_relationship(data1, 'X1', 'X4')

    # Generate all plots
    ts_plot1 = os.path.join(output_dir, "simple_nn_sim1_timeseries.png")
    visualize_time_series(data1, "Simple NN Simulation 1: Linear Chain", save_path=ts_plot1)

    print("\n  Analyzing causality with enhanced MLP...")
    results1 = analyze_all_pairs_mlp(data1)

    causality_plot1 = os.path.join(output_dir, "simple_nn_sim1_causality.png")
    create_mlp_causality_plot(results1, "Simple NN Simulation 1: Linear Chain", save_path=causality_plot1)

    # Create convergence plots
    create_all_convergence_plots(results1, "Simple NN Sim1", output_dir)

    # Save results
    save_results_to_csv(results1, os.path.join(output_dir, "simple_nn_sim1_results.csv"))

    print_mlp_summary(results1, "Simple NN Simulation 1")

    # Simulation 2: Hub + Outsider
    print("\n2. Running Simulation 2: Hub + Outsider")
    print("-" * 50)
    data2 = generate_enhanced_simulation_2(n_points=1200)

    # Debug key relationships
    test_individual_relationship(data2, 'X1', 'X2')
    test_individual_relationship(data2, 'X1', 'X3')
    test_individual_relationship(data2, 'X4', 'X1')

    # Generate all plots
    ts_plot2 = os.path.join(output_dir, "simple_nn_sim2_timeseries.png")
    visualize_time_series(data2, "Simple NN Simulation 2: Hub + Outsider", save_path=ts_plot2)

    print("\n  Analyzing causality with enhanced MLP...")
    results2 = analyze_all_pairs_mlp(data2)

    causality_plot2 = os.path.join(output_dir, "simple_nn_sim2_causality.png")
    create_mlp_causality_plot(results2, "Simple NN Simulation 2: Hub + Outsider", save_path=causality_plot2)

    # Create convergence plots
    create_all_convergence_plots(results2, "Simple NN Sim2", output_dir)

    # Save results
    save_results_to_csv(results2, os.path.join(output_dir, "simple_nn_sim2_results.csv"))

    print_mlp_summary(results2, "Simple NN Simulation 2")

    print(f"\nEnhanced MLP analysis completed!")
    print(f"Check {output_dir} for:")
    print("- Time series plots")
    print("- Causality matrices")
    print("- Convergence plots")
    print("- CSV results")


# Add third simulation for completeness
def generate_enhanced_simulation_3(n_points=1000, strong_strength=0.14, weak_strength=0.05,
                                   delay=3, noise_level=0.35, coupling=0.65, seed=42):
    """ENHANCED Simulation 3: Complex network"""
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    for i in range(max(delay + 1, 4)):
        x1[i] = np.random.normal(0, 0.3)
        x2[i] = np.random.normal(0, 0.3)
        x3[i] = np.random.normal(0, 0.3)
        x4[i] = np.random.normal(0, 0.3)

    for i in range(max(delay + 1, 4), n_points):
        strong_var = 1 + 0.25 * np.sin(0.02 * i)
        weak_var1 = 1 + 0.4 * np.sin(0.03 * i)
        weak_var2 = 1 + 0.6 * np.cos(0.025 * i)

        x1[i] = (0.75 * np.sin(0.085 * i) + 0.25 * np.cos(0.055 * i) +
                 coupling * x1[i - 1] + noise_level * np.random.normal())

        x4[i] = (0.7 * np.cos(0.12 * i) + 0.3 * np.sin(0.08 * i) +
                 coupling * 0.85 * x4[i - 1] + noise_level * np.random.normal())

        # X2 influenced by both X1 and X4
        strong_influence = strong_strength * strong_var * x1[i - delay]
        weak_influence = weak_strength * weak_var2 * x4[i - delay]

        x2[i] = (strong_influence + weak_influence +
                 coupling * x2[i - 1] + noise_level * 1.1 * np.random.normal())

        # X3 weakly influenced by X1
        current_weak1 = weak_strength * weak_var1
        if np.cos(0.04 * i) > 0.2:
            x3[i] = (current_weak1 * x1[i - delay] + coupling * x3[i - 1] +
                     noise_level * 1.4 * np.random.normal())
        else:
            x3[i] = (coupling * x3[i - 1] + noise_level * 1.6 * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def mlp_parameter_sensitivity_analysis(data_dict, var1, var2, sequence_lengths=None, hidden_sizes=None):
    """MLP parameter sensitivity analysis"""
    if sequence_lengths is None:
        sequence_lengths = range(10, 51, 10)
    if hidden_sizes is None:
        hidden_sizes = [(25,), (50,), (50, 25), (100, 50)]

    results_matrix_xy = np.full((len(hidden_sizes), len(sequence_lengths)), np.nan)
    results_matrix_yx = np.full((len(hidden_sizes), len(sequence_lengths)), np.nan)

    print(f"  MLP sensitivity analysis for {var1} ↔ {var2}...")

    for i, hidden_size in enumerate(hidden_sizes):
        for j, seq_len in enumerate(sequence_lengths):
            try:
                min_points = seq_len + 100
                if len(data_dict[var1]) >= min_points:
                    # Test X → Y
                    result_xy = run_mlp_causality(data_dict, var1, var2,
                                                  sequence_length=seq_len, test_split=0.2)
                    results_matrix_xy[i, j] = result_xy['causality_score']

                    # Test Y → X
                    result_yx = run_mlp_causality(data_dict, var2, var1,
                                                  sequence_length=seq_len, test_split=0.2)
                    results_matrix_yx[i, j] = result_yx['causality_score']

            except Exception as e:
                print(f"    Error at hidden={hidden_size}, seq_len={seq_len}: {str(e)[:50]}")
                results_matrix_xy[i, j] = np.nan
                results_matrix_yx[i, j] = np.nan

    return results_matrix_xy, results_matrix_yx, hidden_sizes, list(sequence_lengths)


# Include all simulation and visualization functions from original script
def generate_enhanced_simulation_1(n_points=1000, base_strength=0.12, delay=3,
                                   noise_level=0.35, coupling=0.65, seed=42):
    """ENHANCED Simulation 1: Linear chain - 1 → 2 → 3 → 4"""
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    for i in range(max(delay + 1, 4)):
        x1[i] = np.random.normal(0, 0.3)
        x2[i] = np.random.normal(0, 0.3)
        x3[i] = np.random.normal(0, 0.3)
        x4[i] = np.random.normal(0, 0.3)

    for i in range(max(delay + 1, 4), n_points):
        time_var = 1 + 0.2 * np.sin(0.02 * i)
        current_strength = base_strength * time_var

        x1[i] = (0.7 * np.sin(0.08 * i) + 0.3 * np.cos(0.05 * i) +
                 coupling * x1[i - 1] + noise_level * np.random.normal())

        x2[i] = (current_strength * 0.9 * x1[i - delay] +
                 coupling * x2[i - 1] + noise_level * np.random.normal())

        x3[i] = (current_strength * 0.7 * x2[i - delay] +
                 coupling * x3[i - 1] + noise_level * 1.1 * np.random.normal())

        x4[i] = (current_strength * 0.5 * x3[i - delay] +
                 coupling * x4[i - 1] + noise_level * 1.2 * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


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


def analyze_all_pairs_mlp(data_dict):
    """Updated to use Granger causality"""
    return analyze_all_pairs_granger(data_dict)


def create_mlp_causality_plot(results, title, save_path=None):
    """Updated to use Granger causality"""
    return create_granger_causality_plot(results, title, save_path)


def print_mlp_summary(results, simulation_name):
    """Updated to use Granger causality"""
    return print_granger_summary(results, simulation_name)


def create_mlp_causality_plot(results, title, save_path=None):
    """Create causality matrix plot for MLP results."""
    variables = ['X1', 'X2', 'X3', 'X4']
    n_vars = len(variables)

    causality_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results:
                    result = results[key]
                    if var1 < var2:
                        causality_matrix[i, j] = result[f"{var1}_to_{var2}"]['causality_score']
                        causality_matrix[j, i] = result[f"{var2}_to_{var1}"]['causality_score']

    plt.figure(figsize=(10, 8))
    im = plt.imshow(causality_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(im, label='MLP Causality Score')

    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nMLP Causality Matrix (Row → Column)')

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                color = 'white' if causality_matrix[i, j] > 0.5 else 'black'
                plt.text(j, i, f'{causality_matrix[i, j]:.2f}',
                         ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def print_mlp_summary(results, simulation_name):
    """Print MLP results summary."""
    print(f"\n{'=' * 60}")
    print(f"MLP Summary for {simulation_name}")
    print(f"{'=' * 60}")

    variables = ['X1', 'X2', 'X3', 'X4']

    print("\nDetected Relationships (Score > 0.3):")

    for var1, var2 in combinations(variables, 2):
        key = f"{var1}_{var2}"
        if key in results:
            result = results[key]

            score_xy = result[f"{var1}_to_{var2}"]['causality_score']
            score_yx = result[f"{var2}_to_{var1}"]['causality_score']

            if score_xy > 0.3:
                print(f"  {var1} → {var2}: Score = {score_xy:.3f}")

            if score_yx > 0.3:
                print(f"  {var2} → {var1}: Score = {score_yx:.3f}")


def test_individual_relationship(data_dict, var1, var2):
    """Test and debug individual causality relationship"""
    print(f"\nDEBUG: Testing {var1} → {var2}")

    series1 = np.array(data_dict[var1])
    series2 = np.array(data_dict[var2])

    # Check basic correlation
    correlation = np.corrcoef(series1, series2)[0, 1]
    print(f"  Raw correlation: {correlation:.3f}")

    # Check lagged correlations
    max_lag_corr = 0
    best_lag = 0
    for lag in range(1, 10):
        if len(series1) > lag and len(series2) > lag:
            lag_corr = np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
            if abs(lag_corr) > abs(max_lag_corr):
                max_lag_corr = lag_corr
                best_lag = lag

    print(f"  Best lagged correlation: {max_lag_corr:.3f} at lag {best_lag}")

    # Test causality
    result = run_granger_causality_test(data_dict, var1, var2)
    print(f"  Cross R²: {result['cross_r2']:.3f}")
    print(f"  Self R²: {result['self_r2']:.3f}")
    print(f"  Causality score: {result['causality_score']:.3f}")

    return result


def run_simple_nn_simulations():
    """Main function using scikit-learn MLPRegressor"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"SIMPLE_NN_causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Running Causality Simulations with Scikit-Learn Neural Networks")
    print("=" * 70)
    print(f"Output saved to: {output_dir}")

    # Simulation 1: Linear Chain
    print("\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
    print("-" * 50)
    data1 = generate_enhanced_simulation_1(n_points=1000)

    # Debug individual relationships
    test_individual_relationship(data1, 'X1', 'X2')  # Should be strong
    test_individual_relationship(data1, 'X2', 'X1')  # Should be weak
    test_individual_relationship(data1, 'X1', 'X4')  # Should be weak

    ts_plot1 = os.path.join(output_dir, "simple_nn_sim1_timeseries.png")
    visualize_time_series(data1, "Simple NN Simulation 1: Linear Chain", save_path=ts_plot1)

    print("\n  Analyzing causality with MLP...")
    results1 = analyze_all_pairs_mlp(data1)

    causality_plot1 = os.path.join(output_dir, "simple_nn_sim1_causality.png")
    create_mlp_causality_plot(results1, "Simple NN Simulation 1: Linear Chain", save_path=causality_plot1)

    print_mlp_summary(results1, "Simple NN Simulation 1")

    # Simulation 2: Hub + Outsider
    print("\n2. Running Simulation 2: Hub + Outsider")
    print("-" * 50)
    data2 = generate_enhanced_simulation_2(n_points=1000)

    # Debug key relationships
    test_individual_relationship(data2, 'X1', 'X2')  # Should be strong
    test_individual_relationship(data2, 'X1', 'X3')  # Should be weak
    test_individual_relationship(data2, 'X4', 'X2')  # Should be very weak

    ts_plot2 = os.path.join(output_dir, "simple_nn_sim2_timeseries.png")
    visualize_time_series(data2, "Simple NN Simulation 2: Hub + Outsider", save_path=ts_plot2)

    print("\n  Analyzing causality with MLP...")
    results2 = analyze_all_pairs_mlp(data2)

    causality_plot2 = os.path.join(output_dir, "simple_nn_sim2_causality.png")
    create_mlp_causality_plot(results2, "Simple NN Simulation 2: Hub + Outsider", save_path=causality_plot2)

    print_mlp_summary(results2, "Simple NN Simulation 2")

    print(f"\nSimple NN analysis completed! Check {output_dir}")
    print("Look for non-zero values in causality matrices and debug output above.")


# Add missing simulation functions
def generate_enhanced_simulation_2(n_points=1000, strong_strength=0.15, weak_strength=0.04,
                                   delay=3, noise_level=0.35, coupling=0.65, seed=42):
    """ENHANCED Simulation 2: 1 → 2 (strong), 1 → 3 (weak), 4 is independent"""
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    for i in range(max(delay + 1, 4)):
        x1[i] = np.random.normal(0, 0.3)
        x2[i] = np.random.normal(0, 0.3)
        x3[i] = np.random.normal(0, 0.3)
        x4[i] = np.random.normal(0, 0.3)

    for i in range(max(delay + 1, 4), n_points):
        strong_var = 1 + 0.3 * np.sin(0.015 * i)
        weak_var = 1 + 0.5 * np.sin(0.025 * i)

        x1[i] = (0.8 * np.sin(0.09 * i) + 0.2 * np.cos(0.06 * i) +
                 coupling * x1[i - 1] + noise_level * np.random.normal())

        x2[i] = (strong_strength * strong_var * x1[i - delay] +
                 coupling * x2[i - 1] + noise_level * 0.9 * np.random.normal())

        current_weak = weak_strength * weak_var
        if np.sin(0.03 * i) > 0.3:
            x3[i] = (current_weak * x1[i - delay] + coupling * x3[i - 1] +
                     noise_level * 1.3 * np.random.normal())
        else:
            x3[i] = (current_weak * 0.2 * x1[i - delay] + coupling * x3[i - 1] +
                     noise_level * 1.5 * np.random.normal())

        x4[i] = (0.6 * np.cos(0.11 * i) + 0.4 * np.sin(0.07 * i) +
                 coupling * 0.9 * x4[i - 1] + noise_level * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


if __name__ == "__main__":
    run_simple_nn_simulations()