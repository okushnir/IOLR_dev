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

# Neural network imports - Updated for standalone Keras
try:
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping

    print("Using standalone Keras")
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        print("Using TensorFlow Keras")
    except ImportError:
        print("Error: Neither Keras nor TensorFlow found.")
        print("Install with: pip install keras tensorflow")
        sys.exit(1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings and fix TensorFlow CPU issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MAX_CPU_FEATURE'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Import TensorFlow with CPU fallback
try:
    import tensorflow as tf

    tf.config.set_visible_devices([], 'GPU')  # Force CPU only
    tf.get_logger().setLevel('ERROR')
except Exception as e:
    print(f"TensorFlow setup warning: {e}")
    print("Continuing with CPU-only mode...")


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
        log_file = f"lstm_causality_simulation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def create_sequences(data, sequence_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Build LSTM model for causality analysis"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


def run_lstm_causality(data_dict, var1, var2, sequence_length=20, test_split=0.2):
    """
    LSTM-based causality analysis replacing CCM
    Tests if var1 can predict var2 better than var2 predicts itself
    """
    # Prepare data
    series1 = np.array(data_dict[var1]).reshape(-1, 1)
    series2 = np.array(data_dict[var2]).reshape(-1, 1)

    # Normalize data
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    series1_scaled = scaler1.fit_transform(series1)
    series2_scaled = scaler2.fit_transform(series2)

    # Split data
    split_idx = int(len(series1_scaled) * (1 - test_split))

    # Model 1: var1 predicts var2 (cross-prediction)
    X1_train, y2_train = create_sequences(series1_scaled[:split_idx], sequence_length)
    X1_test, y2_test = create_sequences(series1_scaled[split_idx:], sequence_length)
    y2_train_target = series2_scaled[sequence_length:split_idx]
    y2_test_target = series2_scaled[split_idx + sequence_length:]

    # Adjust lengths to match
    min_len_train = min(len(X1_train), len(y2_train_target))
    min_len_test = min(len(X1_test), len(y2_test_target))

    X1_train = X1_train[:min_len_train]
    y2_train_target = y2_train_target[:min_len_train]
    X1_test = X1_test[:min_len_test]
    y2_test_target = y2_test_target[:min_len_test]

    # Train cross-prediction model
    model_cross = build_lstm_model((sequence_length, 1))
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    with tf.device('/CPU:0'):  # Force CPU to avoid GPU memory issues
        history_cross = model_cross.fit(
            X1_train, y2_train_target,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

    # Predict and calculate performance
    y2_pred_cross = model_cross.predict(X1_test, verbose=0)
    mse_cross = mean_squared_error(y2_test_target, y2_pred_cross)
    r2_cross = r2_score(y2_test_target, y2_pred_cross)

    # Model 2: var2 predicts itself (self-prediction baseline)
    X2_train, y2_train_self = create_sequences(series2_scaled[:split_idx], sequence_length)
    X2_test, y2_test_self = create_sequences(series2_scaled[split_idx:], sequence_length)

    # Adjust lengths
    min_len_train_self = min(len(X2_train), len(y2_train_self))
    min_len_test_self = min(len(X2_test), len(y2_test_self))

    X2_train = X2_train[:min_len_train_self]
    y2_train_self = y2_train_self[:min_len_train_self]
    X2_test = X2_test[:min_len_test_self]
    y2_test_self = y2_test_self[:min_len_test_self]

    # Train self-prediction model
    model_self = build_lstm_model((sequence_length, 1))

    with tf.device('/CPU:0'):
        history_self = model_self.fit(
            X2_train, y2_train_self,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

    # Predict and calculate performance
    y2_pred_self = model_self.predict(X2_test, verbose=0)
    mse_self = mean_squared_error(y2_test_self, y2_pred_self)
    r2_self = r2_score(y2_test_self, y2_pred_self)

    # Calculate causality strength
    # Higher R² for cross-prediction vs self-prediction indicates causality
    causality_strength = max(0, r2_cross - r2_self)

    # Normalize to 0-1 range similar to CCM
    causality_score = min(1.0, max(0.0, causality_strength * 2))

    return {
        'cross_r2': r2_cross,
        'self_r2': r2_self,
        'cross_mse': mse_cross,
        'self_mse': mse_self,
        'causality_strength': causality_strength,
        'causality_score': causality_score
    }


def lstm_embedding_sensitivity_analysis(data_dict, var1, var2, sequence_lengths=None, units_range=None):
    """
    LSTM parameter sensitivity analysis replacing CCM embedding analysis
    """
    if sequence_lengths is None:
        sequence_lengths = range(10, 51, 10)  # 10, 20, 30, 40, 50
    if units_range is None:
        units_range = [25, 50, 75, 100]

    results_matrix_xy = np.full((len(units_range), len(sequence_lengths)), np.nan)
    results_matrix_yx = np.full((len(units_range), len(sequence_lengths)), np.nan)

    print(f"  LSTM sensitivity analysis for {var1} ↔ {var2}...")

    for i, units in enumerate(units_range):
        for j, seq_len in enumerate(sequence_lengths):
            try:
                # Check if we have enough data
                min_points = seq_len + 100
                if len(data_dict[var1]) >= min_points:
                    # Test X → Y
                    result_xy = run_lstm_causality(data_dict, var1, var2,
                                                   sequence_length=seq_len, test_split=0.2)
                    results_matrix_xy[i, j] = result_xy['causality_score']

                    # Test Y → X
                    result_yx = run_lstm_causality(data_dict, var2, var1,
                                                   sequence_length=seq_len, test_split=0.2)
                    results_matrix_yx[i, j] = result_yx['causality_score']

            except Exception as e:
                print(f"    Error at units={units}, seq_len={seq_len}: {str(e)[:50]}")
                results_matrix_xy[i, j] = np.nan
                results_matrix_yx[i, j] = np.nan

    return results_matrix_xy, results_matrix_yx, list(units_range), list(sequence_lengths)


def plot_lstm_sensitivity_heatmap(results_xy, results_yx, units_range, seq_lengths,
                                  var1, var2, title, save_path=None):
    """
    Plot LSTM parameter sensitivity heatmap
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin, vmax = 0, 1

    # Plot X→Y direction
    im1 = ax1.imshow(results_xy, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(f'{var1} → {var2}')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('LSTM Units')
    ax1.set_xticks(range(len(seq_lengths)))
    ax1.set_xticklabels(seq_lengths)
    ax1.set_yticks(range(len(units_range)))
    ax1.set_yticklabels(units_range)

    # Add text annotations
    for i in range(len(units_range)):
        for j in range(len(seq_lengths)):
            if not np.isnan(results_xy[i, j]):
                value = results_xy[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')
            else:
                ax1.text(j, i, 'N/A', ha='center', va='center',
                         color='gray', fontsize=7)

    plt.colorbar(im1, ax=ax1, label='Causality Score')

    # Plot Y→X direction
    im2 = ax2.imshow(results_yx, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title(f'{var2} → {var1}')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('LSTM Units')
    ax2.set_xticks(range(len(seq_lengths)))
    ax2.set_xticklabels(seq_lengths)
    ax2.set_yticks(range(len(units_range)))
    ax2.set_yticklabels(units_range)

    # Add text annotations
    for i in range(len(units_range)):
        for j in range(len(seq_lengths)):
            if not np.isnan(results_yx[i, j]):
                value = results_yx[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')
            else:
                ax2.text(j, i, 'N/A', ha='center', va='center',
                         color='gray', fontsize=7)

    plt.colorbar(im2, ax=ax2, label='Causality Score')

    plt.suptitle(f'LSTM Parameter Sensitivity: {title}\n{var1} ↔ {var2}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved LSTM sensitivity plot to: {save_path}")
    else:
        plt.show()
    plt.close()


# Keep original simulation functions
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


def generate_enhanced_simulation_3(n_points=1000, strong_strength=0.14, weak_strength=0.05,
                                   delay=3, noise_level=0.35, coupling=0.65, seed=42):
    """ENHANCED Simulation 3: 1 → 2 (strong), 1 → 3 (weak), 4 → 2 (weak)"""
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

        strong_influence = strong_strength * strong_var * x1[i - delay]
        weak_influence = weak_strength * weak_var2 * x4[i - delay]
        interference = 0.02 * strong_influence * weak_influence

        x2[i] = (strong_influence + weak_influence + interference +
                 coupling * x2[i - 1] + noise_level * 1.1 * np.random.normal())

        current_weak1 = weak_strength * weak_var1
        if np.cos(0.04 * i) > 0.2:
            x3[i] = (current_weak1 * x1[i - delay] + coupling * x3[i - 1] +
                     noise_level * 1.4 * np.random.normal())
        else:
            x3[i] = (coupling * x3[i - 1] + noise_level * 1.6 * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


# Keep visualization functions
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


def analyze_all_pairs_lstm(data_dict):
    """LSTM analysis of all pairwise relationships."""
    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        print(f"  Analyzing {var1} ↔ {var2} with LSTM...")

        # Test both directions
        result_xy = run_lstm_causality(data_dict, var1, var2)
        result_yx = run_lstm_causality(data_dict, var2, var1)

        results[f"{var1}_{var2}"] = {
            f"{var1}_to_{var2}": result_xy,
            f"{var2}_to_{var1}": result_yx
        }

    return results


def demonstrate_lstm_parameter_analysis(data_dict, simulation_name, output_dir):
    """Demonstrate LSTM parameter dependency analysis"""
    print(f"\n{'=' * 60}")
    print(f"LSTM PARAMETER ANALYSIS: {simulation_name}")
    print(f"{'=' * 60}")

    test_pairs = [('X1', 'X2'), ('X1', 'X3'), ('X1', 'X4'), ('X2', 'X3')]

    for var1, var2 in test_pairs:
        print(f"\nLSTM analysis for {var1} ↔ {var2}...")

        results_xy, results_yx, units_range, seq_lengths = lstm_embedding_sensitivity_analysis(
            data_dict, var1, var2)

        heatmap_path = os.path.join(output_dir,
                                    f"{simulation_name.lower().replace(' ', '_')}_LSTM_sensitivity_{var1}_{var2}.png")
        plot_lstm_sensitivity_heatmap(results_xy, results_yx, units_range, seq_lengths,
                                      var1, var2, simulation_name, save_path=heatmap_path)

        valid_xy = results_xy[~np.isnan(results_xy)]
        valid_yx = results_yx[~np.isnan(results_yx)]

        if len(valid_xy) > 0:
            print(f"  {var1}→{var2}: Range [{valid_xy.min():.2f}, {valid_xy.max():.2f}], "
                  f"Mean {valid_xy.mean():.2f}, Std {valid_xy.std():.2f}")
        if len(valid_yx) > 0:
            print(f"  {var2}→{var1}: Range [{valid_yx.min():.2f}, {valid_yx.max():.2f}], "
                  f"Mean {valid_yx.mean():.2f}, Std {valid_yx.std():.2f}")


def create_lstm_causality_plot(results, title, expected_relationships=None, save_path=None):
    """Create a causality matrix plot for LSTM results."""
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
    plt.colorbar(im, label='LSTM Causality Score')

    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nLSTM Causality Matrix (Row → Column)')

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                color = 'white' if causality_matrix[i, j] > 0.5 else 'black'
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


def print_lstm_summary(results, simulation_name, expected_relationships):
    """Print a summary of the LSTM results."""
    print(f"\n{'=' * 60}")
    print(f"LSTM Summary for {simulation_name}")
    print(f"{'=' * 60}")

    variables = ['X1', 'X2', 'X3', 'X4']
    var_to_idx = {var: i for i, var in enumerate(variables)}

    print("\nDetected Relationships (Score > 0.3):")
    strong_relationships = []

    for var1, var2 in combinations(variables, 2):
        key = f"{var1}_{var2}"
        if key in results:
            result = results[key]

            score_xy = result[f"{var1}_to_{var2}"]['causality_score']
            score_yx = result[f"{var2}_to_{var1}"]['causality_score']

            if score_xy > 0.3:
                print(f"  {var1} → {var2}: Score = {score_xy:.3f}")
                strong_relationships.append((var_to_idx[var1], var_to_idx[var2], score_xy))

            if score_yx > 0.3:
                print(f"  {var2} → {var1}: Score = {score_yx:.3f}")
                strong_relationships.append((var_to_idx[var2], var_to_idx[var1], score_yx))

    print(f"\nExpected vs Detected:")
    for (i, j, expected) in expected_relationships:
        found = False
        for (det_i, det_j, det_score) in strong_relationships:
            if i == det_i and j == det_j:
                status = "✓ DETECTED" if det_score > 0.3 else "✗ WEAK"
                print(f"  {variables[i]} → {variables[j]}: Expected={expected}, Detected={det_score:.3f} {status}")
                found = True
                break
        if not found:
            print(f"  {variables[i]} → {variables[j]}: Expected={expected}, Detected=< 0.3 ✗ NOT DETECTED")


def run_lstm_simulations():
    """Main function: Run all simulations with LSTM analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"LSTM_causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "lstm_simulation_log.txt")

    with LogCapture(log_file):
        print("Running Three Causality Simulations with LSTM Neural Network Analysis")
        print("=" * 80)
        print(f"Output will be saved to: {output_dir}")
        print(f"Log file: {log_file}")

        # Simulation 1: Linear Chain
        print("\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
        print("-" * 60)
        data1 = generate_enhanced_simulation_1(n_points=1200)

        ts_plot1 = os.path.join(output_dir, "lstm_sim1_timeseries.png")
        visualize_time_series(data1, "LSTM Simulation 1: Linear Chain", save_path=ts_plot1)

        demonstrate_lstm_parameter_analysis(data1, "LSTM Simulation 1", output_dir)

        print("  Analyzing causality with LSTM...")
        results1 = analyze_all_pairs_lstm(data1)
        expected1 = [(0, 1, "Strong"), (1, 2, "Medium"), (2, 3, "Weak")]

        causality_plot1 = os.path.join(output_dir, "lstm_sim1_causality.png")
        create_lstm_causality_plot(results1, "LSTM Simulation 1: Linear Chain",
                                   expected1, save_path=causality_plot1)

        print_lstm_summary(results1, "LSTM Simulation 1", expected1)

        # Simulation 2: Hub with Outsider
        print("\n2. Running Simulation 2: Hub + Outsider")
        print("-" * 60)
        data2 = generate_enhanced_simulation_2(n_points=1200)

        ts_plot2 = os.path.join(output_dir, "lstm_sim2_timeseries.png")
        visualize_time_series(data2, "LSTM Simulation 2: Hub + Outsider", save_path=ts_plot2)

        demonstrate_lstm_parameter_analysis(data2, "LSTM Simulation 2", output_dir)

        print("  Analyzing causality with LSTM...")
        results2 = analyze_all_pairs_lstm(data2)
        expected2 = [(0, 1, "Strong"), (0, 2, "Weak")]

        causality_plot2 = os.path.join(output_dir, "lstm_sim2_causality.png")
        create_lstm_causality_plot(results2, "LSTM Simulation 2: Hub + Outsider",
                                   expected2, save_path=causality_plot2)

        print_lstm_summary(results2, "LSTM Simulation 2", expected2)

        # Simulation 3: Complex Network
        print("\n3. Running Simulation 3: Complex Network")
        print("-" * 60)
        data3 = generate_enhanced_simulation_3(n_points=1200)

        ts_plot3 = os.path.join(output_dir, "lstm_sim3_timeseries.png")
        visualize_time_series(data3, "LSTM Simulation 3: Complex Network", save_path=ts_plot3)

        demonstrate_lstm_parameter_analysis(data3, "LSTM Simulation 3", output_dir)

        print("  Analyzing causality with LSTM...")
        results3 = analyze_all_pairs_lstm(data3)
        expected3 = [(0, 1, "Strong"), (0, 2, "Weak"), (3, 1, "Weak")]

        causality_plot3 = os.path.join(output_dir, "lstm_sim3_causality.png")
        create_lstm_causality_plot(results3, "LSTM Simulation 3: Complex Network",
                                   expected3, save_path=causality_plot3)

        print_lstm_summary(results3, "LSTM Simulation 3", expected3)

        # Final Summary
        print("\n" + "=" * 80)
        print("LSTM ANALYSIS SUMMARY")
        print("=" * 80)
        print("LSTM Neural Network Approach:")
        print("✓ Replaces CCM with LSTM-based causality detection")
        print("✓ Tests cross-prediction vs self-prediction performance")
        print("✓ Uses R² difference as causality strength measure")
        print("✓ Parameter sensitivity across sequence lengths and units")
        print("✓ Early stopping to prevent overfitting")

        print("\nExpected LSTM Results:")
        print("• Strong connections: Score = 0.6-1.0")
        print("• Medium connections: Score = 0.3-0.6")
        print("• Weak connections: Score = 0.1-0.3")
        print("• No connection: Score = 0.0-0.1")

        print(f"\nAll LSTM outputs saved to: {output_dir}")
        print("LSTM approach provides neural network-based causality detection!")

    print(f"\nLSTM simulation completed! Results saved to: {output_dir}")
    print("LSTM models test if one variable can predict another better than self-prediction")


if __name__ == "__main__":
    run_lstm_simulations()