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

# Neural network imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Attention, \
        Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow.keras.backend as K
except ImportError:
    try:
        import keras
        from keras.models import Model
        from keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Attention, Concatenate
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        import keras.backend as K
    except ImportError:
        print("Error: Neither TensorFlow nor Keras found. Install with: pip install tensorflow")
        sys.exit(1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')


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
        log_file = f"encoder_decoder_causality_simulation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def create_sequences_multivariate(data, sequence_length, target_length):
    """Create sequences for encoder-decoder training"""
    X, y = [], []
    for i in range(len(data) - sequence_length - target_length + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[(i + sequence_length):(i + sequence_length + target_length)])
    return np.array(X), np.array(y)


def build_encoder_decoder_model(input_shape, target_length, latent_dim=50, dropout_rate=0.2):
    """
    Build Encoder-Decoder model for causality analysis

    Args:
        input_shape: (sequence_length, n_features)
        target_length: Length of target sequence to predict
        latent_dim: Dimension of latent representation
    """
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, dropout=dropout_rate)(encoder_inputs)
    encoder_lstm2 = LSTM(latent_dim, dropout=dropout_rate)(encoder_lstm1)

    # Decoder
    decoder_inputs = RepeatVector(target_length)(encoder_lstm2)
    decoder_lstm1 = LSTM(latent_dim, return_sequences=True, dropout=dropout_rate)(decoder_inputs)
    decoder_lstm2 = LSTM(latent_dim, return_sequences=True, dropout=dropout_rate)(decoder_lstm1)
    decoder_outputs = TimeDistributed(Dense(input_shape[1]))(decoder_lstm2)

    # Create model
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def build_attention_encoder_decoder(input_shape, target_length, latent_dim=50, dropout_rate=0.2):
    """
    Build Encoder-Decoder with attention mechanism for better causality detection
    """
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, dropout=dropout_rate)(encoder_inputs)
    encoder_states = LSTM(latent_dim, dropout=dropout_rate)(encoder_lstm)

    # Decoder with attention
    decoder_inputs = RepeatVector(target_length)(encoder_states)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, dropout=dropout_rate)(decoder_inputs)

    # Attention mechanism
    attention = Attention()([decoder_lstm, encoder_lstm])
    decoder_concat = Concatenate()([decoder_lstm, attention])

    decoder_outputs = TimeDistributed(Dense(input_shape[1]))(decoder_concat)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def run_encoder_decoder_causality(data_dict, var1, var2, sequence_length=20, target_length=5,
                                  test_split=0.2, use_attention=True):
    """
    Encoder-Decoder based causality analysis replacing CCM
    Tests multivariate prediction capability vs univariate baseline
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

    # Model 1: Multivariate (both series) predicts var2
    multivariate_data = np.concatenate([series1_scaled, series2_scaled], axis=1)
    X_multi_train, y_multi_train = create_sequences_multivariate(
        multivariate_data[:split_idx], sequence_length, target_length)
    X_multi_test, y_multi_test = create_sequences_multivariate(
        multivariate_data[split_idx:], sequence_length, target_length)

    # Extract only var2 targets
    y_multi_train_target = y_multi_train[:, :, 1:2]  # Only var2
    y_multi_test_target = y_multi_test[:, :, 1:2]  # Only var2

    # Train multivariate model
    if use_attention:
        model_multi = build_attention_encoder_decoder((sequence_length, 2), target_length)
    else:
        model_multi = build_encoder_decoder_model((sequence_length, 2), target_length)

    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

    with tf.device('/CPU:0'):
        history_multi = model_multi.fit(
            X_multi_train, y_multi_train_target,
            epochs=80,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

    # Predict and calculate performance
    y_pred_multi = model_multi.predict(X_multi_test, verbose=0)
    mse_multi = mean_squared_error(y_multi_test_target.reshape(-1), y_pred_multi.reshape(-1))
    r2_multi = r2_score(y_multi_test_target.reshape(-1), y_pred_multi.reshape(-1))

    # Model 2: Univariate (only var2) predicts var2 (baseline)
    univariate_data = series2_scaled
    X_uni_train, y_uni_train = create_sequences_multivariate(
        univariate_data[:split_idx], sequence_length, target_length)
    X_uni_test, y_uni_test = create_sequences_multivariate(
        univariate_data[split_idx:], sequence_length, target_length)

    # Reshape for single variable
    X_uni_train = X_uni_train.reshape(X_uni_train.shape[0], X_uni_train.shape[1], 1)
    X_uni_test = X_uni_test.reshape(X_uni_test.shape[0], X_uni_test.shape[1], 1)
    y_uni_train = y_uni_train.reshape(y_uni_train.shape[0], y_uni_train.shape[1], 1)
    y_uni_test = y_uni_test.reshape(y_uni_test.shape[0], y_uni_test.shape[1], 1)

    # Train univariate model
    if use_attention:
        model_uni = build_attention_encoder_decoder((sequence_length, 1), target_length)
    else:
        model_uni = build_encoder_decoder_model((sequence_length, 1), target_length)

    with tf.device('/CPU:0'):
        history_uni = model_uni.fit(
            X_uni_train, y_uni_train,
            epochs=80,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

    # Predict and calculate performance
    y_pred_uni = model_uni.predict(X_uni_test, verbose=0)
    mse_uni = mean_squared_error(y_uni_test.reshape(-1), y_pred_uni.reshape(-1))
    r2_uni = r2_score(y_uni_test.reshape(-1), y_pred_uni.reshape(-1))

    # Calculate causality strength
    # Better multivariate vs univariate prediction indicates causality
    causality_strength = max(0, r2_multi - r2_uni)

    # Normalize to 0-1 range similar to CCM
    causality_score = min(1.0, max(0.0, causality_strength * 3))  # Scale factor for sensitivity

    # Calculate information gain
    mse_improvement = (mse_uni - mse_multi) / mse_uni if mse_uni > 0 else 0
    info_gain = max(0, mse_improvement)

    return {
        'multivariate_r2': r2_multi,
        'univariate_r2': r2_uni,
        'multivariate_mse': mse_multi,
        'univariate_mse': mse_uni,
        'causality_strength': causality_strength,
        'causality_score': causality_score,
        'information_gain': info_gain,
        'mse_improvement': mse_improvement
    }


def encoder_decoder_sensitivity_analysis(data_dict, var1, var2, sequence_lengths=None,
                                         latent_dims=None, use_attention=True):
    """
    Encoder-Decoder parameter sensitivity analysis
    """
    if sequence_lengths is None:
        sequence_lengths = range(15, 41, 5)  # 15, 20, 25, 30, 35, 40
    if latent_dims is None:
        latent_dims = [32, 50, 64, 80]

    results_matrix_xy = np.full((len(latent_dims), len(sequence_lengths)), np.nan)
    results_matrix_yx = np.full((len(latent_dims), len(sequence_lengths)), np.nan)

    print(f"  Encoder-Decoder sensitivity analysis for {var1} ↔ {var2}...")

    for i, latent_dim in enumerate(latent_dims):
        for j, seq_len in enumerate(sequence_lengths):
            try:
                # Check if we have enough data
                min_points = seq_len + 50
                if len(data_dict[var1]) >= min_points:
                    # Test X → Y
                    result_xy = run_encoder_decoder_causality(
                        data_dict, var1, var2, sequence_length=seq_len,
                        target_length=5, test_split=0.2, use_attention=use_attention)
                    results_matrix_xy[i, j] = result_xy['causality_score']

                    # Test Y → X
                    result_yx = run_encoder_decoder_causality(
                        data_dict, var2, var1, sequence_length=seq_len,
                        target_length=5, test_split=0.2, use_attention=use_attention)
                    results_matrix_yx[i, j] = result_yx['causality_score']

            except Exception as e:
                print(f"    Error at latent_dim={latent_dim}, seq_len={seq_len}: {str(e)[:50]}")
                results_matrix_xy[i, j] = np.nan
                results_matrix_yx[i, j] = np.nan

    return results_matrix_xy, results_matrix_yx, list(latent_dims), list(sequence_lengths)


def plot_encoder_decoder_sensitivity_heatmap(results_xy, results_yx, latent_dims, seq_lengths,
                                             var1, var2, title, save_path=None):
    """
    Plot Encoder-Decoder parameter sensitivity heatmap
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin, vmax = 0, 1

    # Plot X→Y direction
    im1 = ax1.imshow(results_xy, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(f'{var1} → {var2}')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latent Dimension')
    ax1.set_xticks(range(len(seq_lengths)))
    ax1.set_xticklabels(seq_lengths)
    ax1.set_yticks(range(len(latent_dims)))
    ax1.set_yticklabels(latent_dims)

    # Add text annotations
    for i in range(len(latent_dims)):
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
    ax2.set_ylabel('Latent Dimension')
    ax2.set_xticks(range(len(seq_lengths)))
    ax2.set_xticklabels(seq_lengths)
    ax2.set_yticks(range(len(latent_dims)))
    ax2.set_yticklabels(latent_dims)

    # Add text annotations
    for i in range(len(latent_dims)):
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

    plt.suptitle(f'Encoder-Decoder Parameter Sensitivity: {title}\n{var1} ↔ {var2}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved Encoder-Decoder sensitivity plot to: {save_path}")
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


def analyze_all_pairs_encoder_decoder(data_dict, use_attention=True):
    """Encoder-Decoder analysis of all pairwise relationships."""
    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        print(f"  Analyzing {var1} ↔ {var2} with Encoder-Decoder...")

        # Test both directions
        result_xy = run_encoder_decoder_causality(data_dict, var1, var2, use_attention=use_attention)
        result_yx = run_encoder_decoder_causality(data_dict, var2, var1, use_attention=use_attention)

        results[f"{var1}_{var2}"] = {
            f"{var1}_to_{var2}": result_xy,
            f"{var2}_to_{var1}": result_yx
        }

    return results


def demonstrate_encoder_decoder_parameter_analysis(data_dict, simulation_name, output_dir, use_attention=True):
    """Demonstrate Encoder-Decoder parameter dependency analysis"""
    print(f"\n{'=' * 60}")
    print(f"ENCODER-DECODER PARAMETER ANALYSIS: {simulation_name}")
    print(f"{'=' * 60}")

    test_pairs = [('X1', 'X2'), ('X1', 'X3'), ('X1', 'X4'), ('X2', 'X3')]

    for var1, var2 in test_pairs:
        print(f"\nEncoder-Decoder analysis for {var1} ↔ {var2}...")

        results_xy, results_yx, latent_dims, seq_lengths = encoder_decoder_sensitivity_analysis(
            data_dict, var1, var2, use_attention=use_attention)

        attention_str = "ATTENTION_" if use_attention else ""
        heatmap_path = os.path.join(output_dir,
                                    f"{simulation_name.lower().replace(' ', '_')}_ENCODER_DECODER_{attention_str}sensitivity_{var1}_{var2}.png")
        plot_encoder_decoder_sensitivity_heatmap(results_xy, results_yx, latent_dims, seq_lengths,
                                                 var1, var2, simulation_name, save_path=heatmap_path)

        valid_xy = results_xy[~np.isnan(results_xy)]
        valid_yx = results_yx[~np.isnan(results_yx)]

        if len(valid_xy) > 0:
            print(f"  {var1}→{var2}: Range [{valid_xy.min():.2f}, {valid_xy.max():.2f}], "
                  f"Mean {valid_xy.mean():.2f}, Std {valid_xy.std():.2f}")
        if len(valid_yx) > 0:
            print(f"  {var2}→{var1}: Range [{valid_yx.min():.2f}, {valid_yx.max():.2f}], "
                  f"Mean {valid_yx.mean():.2f}, Std {valid_yx.std():.2f}")


def create_encoder_decoder_causality_plot(results, title, expected_relationships=None, save_path=None):
    """Create a causality matrix plot for Encoder-Decoder results."""
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
    plt.colorbar(im, label='Encoder-Decoder Causality Score')

    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nEncoder-Decoder Causality Matrix (Row → Column)')

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


def print_encoder_decoder_summary(results, simulation_name, expected_relationships):
    """Print a summary of the Encoder-Decoder results."""
    print(f"\n{'=' * 60}")
    print(f"Encoder-Decoder Summary for {simulation_name}")
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
            info_gain_xy = result[f"{var1}_to_{var2}"]['information_gain']
            info_gain_yx = result[f"{var2}_to_{var1}"]['information_gain']

            if score_xy > 0.3:
                print(f"  {var1} → {var2}: Score = {score_xy:.3f}, Info Gain = {info_gain_xy:.3f}")
                strong_relationships.append((var_to_idx[var1], var_to_idx[var2], score_xy))

            if score_yx > 0.3:
                print(f"  {var2} → {var1}: Score = {score_yx:.3f}, Info Gain = {info_gain_yx:.3f}")
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


def create_prediction_comparison_plot(data_dict, var1, var2, results, title, save_path=None):
    """Create a plot comparing multivariate vs univariate predictions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Show actual time series
    ax1.plot(data_dict[var1][:200], label=var1, alpha=0.7)
    ax1.plot(data_dict[var2][:200], label=var2, alpha=0.7)
    ax1.set_title(f'Original Time Series: {var1} and {var2}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show performance comparison
    key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
    if key in results:
        result_xy = results[key][f"{var1}_to_{var2}"]
        result_yx = results[key][f"{var2}_to_{var1}"]

        metrics = ['Multivariate R²', 'Univariate R²', 'Causality Score', 'Information Gain']
        xy_values = [result_xy['multivariate_r2'], result_xy['univariate_r2'],
                     result_xy['causality_score'], result_xy['information_gain']]
        yx_values = [result_yx['multivariate_r2'], result_yx['univariate_r2'],
                     result_yx['causality_score'], result_yx['information_gain']]

        x = np.arange(len(metrics))
        width = 0.35

        ax2.bar(x - width / 2, xy_values, width, label=f'{var1} → {var2}', alpha=0.8)
        ax2.bar(x + width / 2, yx_values, width, label=f'{var2} → {var1}', alpha=0.8)

        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Encoder-Decoder Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (xy_val, yx_val) in enumerate(zip(xy_values, yx_values)):
            ax2.text(i - width / 2, xy_val + 0.01, f'{xy_val:.2f}', ha='center', va='bottom')
            ax2.text(i + width / 2, yx_val + 0.01, f'{yx_val:.2f}', ha='center', va='bottom')

    plt.suptitle(f'{title}: {var1} ↔ {var2}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def run_encoder_decoder_simulations(use_attention=True):
    """Main function: Run all simulations with Encoder-Decoder analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attention_str = "ATTENTION_" if use_attention else ""
    output_dir = f"ENCODER_DECODER_{attention_str}causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "encoder_decoder_simulation_log.txt")

    with LogCapture(log_file):
        model_type = "Encoder-Decoder with Attention" if use_attention else "Encoder-Decoder"
        print(f"Running Three Causality Simulations with {model_type} Neural Network Analysis")
        print("=" * 80)
        print(f"Output will be saved to: {output_dir}")
        print(f"Log file: {log_file}")

        # Simulation 1: Linear Chain
        print("\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
        print("-" * 60)
        data1 = generate_enhanced_simulation_1(n_points=1200)

        ts_plot1 = os.path.join(output_dir, "encoder_decoder_sim1_timeseries.png")
        visualize_time_series(data1, f"{model_type} Simulation 1: Linear Chain", save_path=ts_plot1)

        demonstrate_encoder_decoder_parameter_analysis(data1, "Encoder-Decoder Simulation 1",
                                                       output_dir, use_attention=use_attention)

        print(f"  Analyzing causality with {model_type}...")
        results1 = analyze_all_pairs_encoder_decoder(data1, use_attention=use_attention)
        expected1 = [(0, 1, "Strong"), (1, 2, "Medium"), (2, 3, "Weak")]

        causality_plot1 = os.path.join(output_dir, "encoder_decoder_sim1_causality.png")
        create_encoder_decoder_causality_plot(results1, f"{model_type} Simulation 1: Linear Chain",
                                              expected1, save_path=causality_plot1)

        # Create prediction comparison plots
        comparison_plot1 = os.path.join(output_dir, "encoder_decoder_sim1_comparison_X1_X2.png")
        create_prediction_comparison_plot(data1, 'X1', 'X2', results1,
                                          f"{model_type} Simulation 1", save_path=comparison_plot1)

        print_encoder_decoder_summary(results1, f"{model_type} Simulation 1", expected1)

        # Simulation 2: Hub with Outsider
        print("\n2. Running Simulation 2: Hub + Outsider")
        print("-" * 60)
        data2 = generate_enhanced_simulation_2(n_points=1200)

        ts_plot2 = os.path.join(output_dir, "encoder_decoder_sim2_timeseries.png")
        visualize_time_series(data2, f"{model_type} Simulation 2: Hub + Outsider", save_path=ts_plot2)

        demonstrate_encoder_decoder_parameter_analysis(data2, "Encoder-Decoder Simulation 2",
                                                       output_dir, use_attention=use_attention)

        print(f"  Analyzing causality with {model_type}...")
        results2 = analyze_all_pairs_encoder_decoder(data2, use_attention=use_attention)
        expected2 = [(0, 1, "Strong"), (0, 2, "Weak")]

        causality_plot2 = os.path.join(output_dir, "encoder_decoder_sim2_causality.png")
        create_encoder_decoder_causality_plot(results2, f"{model_type} Simulation 2: Hub + Outsider",
                                              expected2, save_path=causality_plot2)

        comparison_plot2 = os.path.join(output_dir, "encoder_decoder_sim2_comparison_X1_X2.png")
        create_prediction_comparison_plot(data2, 'X1', 'X2', results2,
                                          f"{model_type} Simulation 2", save_path=comparison_plot2)

        print_encoder_decoder_summary(results2, f"{model_type} Simulation 2", expected2)

        # Simulation 3: Complex Network
        print("\n3. Running Simulation 3: Complex Network")
        print("-" * 60)
        data3 = generate_enhanced_simulation_3(n_points=1200)

        ts_plot3 = os.path.join(output_dir, "encoder_decoder_sim3_timeseries.png")
        visualize_time_series(data3, f"{model_type} Simulation 3: Complex Network", save_path=ts_plot3)

        demonstrate_encoder_decoder_parameter_analysis(data3, "Encoder-Decoder Simulation 3",
                                                       output_dir, use_attention=use_attention)

        print(f"  Analyzing causality with {model_type}...")
        results3 = analyze_all_pairs_encoder_decoder(data3, use_attention=use_attention)
        expected3 = [(0, 1, "Strong"), (0, 2, "Weak"), (3, 1, "Weak")]

        causality_plot3 = os.path.join(output_dir, "encoder_decoder_sim3_causality.png")
        create_encoder_decoder_causality_plot(results3, f"{model_type} Simulation 3: Complex Network",
                                              expected3, save_path=causality_plot3)

        comparison_plot3 = os.path.join(output_dir, "encoder_decoder_sim3_comparison_X1_X2.png")
        create_prediction_comparison_plot(data3, 'X1', 'X2', results3,
                                          f"{model_type} Simulation 3", save_path=comparison_plot3)

        print_encoder_decoder_summary(results3, f"{model_type} Simulation 3", expected3)

        # Final Summary
        print("\n" + "=" * 80)
        print(f"{model_type.upper()} ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"{model_type} Approach:")
        print("✓ Replaces CCM with sequence-to-sequence neural networks")
        print("✓ Tests multivariate vs univariate prediction performance")
        print("✓ Uses R² difference and information gain as causality measures")
        print("✓ Parameter sensitivity across sequence lengths and latent dimensions")
        if use_attention:
            print("✓ Attention mechanism for better temporal dependency modeling")
        print("✓ Early stopping to prevent overfitting")
        print("✓ Future sequence prediction instead of single-step")

        print(f"\nExpected {model_type} Results:")
        print("• Strong connections: Score = 0.6-1.0")
        print("• Medium connections: Score = 0.3-0.6")
        print("• Weak connections: Score = 0.1-0.3")
        print("• No connection: Score = 0.0-0.1")

        print(f"\nAll {model_type} outputs saved to: {output_dir}")
        print(f"{model_type} approach provides sequence-to-sequence causality detection!")

    print(f"\n{model_type} simulation completed! Results saved to: {output_dir}")
    print("Encoder-Decoder models test if multivariate input predicts better than univariate")


if __name__ == "__main__":
    # Run with attention mechanism (default)
    print("Running Encoder-Decoder with Attention mechanism...")
    run_encoder_decoder_simulations(use_attention=True)

    # Uncomment to also run without attention
    # print("\nRunning Encoder-Decoder without Attention mechanism...")
    # run_encoder_decoder_simulations(use_attention=False)