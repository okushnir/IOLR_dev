import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import warnings
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import contextlib
import sys
import logging

# Comprehensive warning suppression
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Try TensorFlow imports with fallbacks
try:
    try:
        import tensorflow as tf
        print("TensorFlow imported successfully!")
        print(f"Version: {tf.__version__}")
    except Exception as e:
        print(f"TensorFlow import failed: {e}")
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("TensorFlow not available. Using sklearn-based embeddings...")
    TENSORFLOW_AVAILABLE = False


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


class LSTMEmbedder:
    """LSTM-based time series embedder that learns nonlinear embeddings"""

    def __init__(self, embedding_dim=3, sequence_length=10, hidden_units=32):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Use SimplifiedEmbedder instead.")

        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _create_sequences(self, data):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def _build_model(self):
        """Build LSTM autoencoder for embedding learning"""
        encoder_input = keras.Input(shape=(self.sequence_length, 1))

        lstm1 = layers.LSTM(self.hidden_units, return_sequences=True)(encoder_input)
        lstm1 = layers.Dropout(0.2)(lstm1)
        lstm2 = layers.LSTM(self.hidden_units // 2, return_sequences=False)(lstm1)
        lstm2 = layers.Dropout(0.2)(lstm2)

        embedding = layers.Dense(self.embedding_dim, activation='tanh', name='embedding')(lstm2)

        decoder_input = layers.RepeatVector(self.sequence_length)(embedding)
        lstm3 = layers.LSTM(self.hidden_units // 2, return_sequences=True)(decoder_input)
        lstm3 = layers.Dropout(0.2)(lstm3)
        lstm4 = layers.LSTM(self.hidden_units, return_sequences=True)(lstm3)
        decoder_output = layers.TimeDistributed(layers.Dense(1))(lstm4)

        autoencoder = keras.Model(encoder_input, decoder_output)
        autoencoder.compile(optimizer='adam', loss='mse')

        encoder = keras.Model(encoder_input, embedding)

        return autoencoder, encoder

    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2, verbose=0):
        """Fit LSTM embedder to time series data"""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

        X, y = self._create_sequences(data_scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y_seq = np.array([X[i] for i in range(len(X))])

        self.model, self.encoder = self._build_model()

        early_stopping = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor='val_loss'
        )

        history = self.model.fit(
            X, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )

        self.is_fitted = True
        return history

    def embed(self, data):
        """Generate LSTM embeddings for time series"""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before embedding")

        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()

        embeddings = []

        for i in range(len(data_scaled) - self.sequence_length + 1):
            seq = data_scaled[i:i + self.sequence_length]
            seq_reshaped = seq.reshape(1, self.sequence_length, 1)
            embedding = self.encoder.predict(seq_reshaped, verbose=0)[0]
            embeddings.append(embedding)

        return np.array(embeddings)


class SimplifiedEmbedder:
    """Simplified embedder using PCA when TensorFlow is not available"""

    def __init__(self, embedding_dim=3, sequence_length=10, method='pca'):
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.method = method
        self.scaler = StandardScaler()
        self.embedder = None
        self.is_fitted = False

    def _create_sequences(self, data):
        """Create delay embedding sequences"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def fit(self, data, **kwargs):
        """Fit embedder using PCA"""
        sequences = self._create_sequences(data)
        sequences_scaled = self.scaler.fit_transform(sequences)

        self.embedder = PCA(n_components=self.embedding_dim)
        self.embedder.fit(sequences_scaled)
        self.is_fitted = True
        return None

    def embed(self, data):
        """Generate embeddings"""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted first")

        sequences = self._create_sequences(data)
        sequences_scaled = self.scaler.transform(sequences)
        embeddings = self.embedder.transform(sequences_scaled)

        return embeddings


class LSTMCCM:
    """CCM implementation using LSTM/PCA embeddings instead of delay embeddings"""

    def __init__(self, embedding_dim=3, sequence_length=10, n_neighbors=5):
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.n_neighbors = n_neighbors

        if TENSORFLOW_AVAILABLE:
            self.embedder_x = LSTMEmbedder(embedding_dim, sequence_length)
            self.embedder_y = LSTMEmbedder(embedding_dim, sequence_length)
        else:
            self.embedder_x = SimplifiedEmbedder(embedding_dim, sequence_length, method='pca')
            self.embedder_y = SimplifiedEmbedder(embedding_dim, sequence_length, method='pca')

    def fit_embedders(self, x_data, y_data, epochs=50, verbose=0):
        """Fit embedders to both time series"""
        embedder_type = "LSTM" if TENSORFLOW_AVAILABLE else "PCA"
        print(
            f"  Training {embedder_type} embedder for X (dim={self.embedding_dim}, seq_len={self.sequence_length})...")

        if TENSORFLOW_AVAILABLE:
            self.embedder_x.fit(x_data, epochs=epochs, verbose=verbose)
            print(f"  Training {embedder_type} embedder for Y...")
            self.embedder_y.fit(y_data, epochs=epochs, verbose=verbose)
        else:
            self.embedder_x.fit(x_data)
            print(f"  Training {embedder_type} embedder for Y...")
            self.embedder_y.fit(y_data)

    def cross_map(self, x_data, y_data, lib_sizes=None, n_bootstrap=20):
        """Perform CCM using LSTM/PCA embeddings"""
        if lib_sizes is None:
            lib_sizes = [50, 100, 150, 200]

        embedder_type = "LSTM" if TENSORFLOW_AVAILABLE else "PCA"
        print(f"  Generating {embedder_type} embeddings...")
        x_embedded = self.embedder_x.embed(x_data)
        y_embedded = self.embedder_y.embed(y_data)

        min_len = min(len(x_embedded), len(y_embedded))
        x_embedded = x_embedded[:min_len]
        y_embedded = y_embedded[:min_len]
        y_aligned = y_data[self.sequence_length - 1:self.sequence_length - 1 + min_len]
        x_aligned = x_data[self.sequence_length - 1:self.sequence_length - 1 + min_len]

        results = []

        for lib_size in lib_sizes:
            if lib_size > min_len:
                continue

            x_to_y_scores = []
            y_to_x_scores = []

            for _ in range(n_bootstrap):
                lib_indices = np.random.choice(min_len, lib_size, replace=False)

                rho_xy = self._single_cross_map(
                    y_embedded[lib_indices], y_aligned[lib_indices],
                    y_embedded, x_aligned
                )
                x_to_y_scores.append(rho_xy)

                rho_yx = self._single_cross_map(
                    x_embedded[lib_indices], x_aligned[lib_indices],
                    x_embedded, y_aligned
                )
                y_to_x_scores.append(rho_yx)

            results.append({
                'LibSize': lib_size,
                'X:Y': np.mean(x_to_y_scores),
                'Y:X': np.mean(y_to_x_scores),
                'X:Y_std': np.std(x_to_y_scores),
                'Y:X_std': np.std(y_to_x_scores)
            })

        return pd.DataFrame(results)

    def _single_cross_map(self, lib_embeddings, lib_targets, pred_embeddings, actual_targets):
        """Single cross-mapping calculation using nearest neighbors in embedding space"""
        if len(lib_embeddings) < self.n_neighbors:
            return 0.0

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nbrs.fit(lib_embeddings)

        predictions = []
        actuals = []

        for i, query_point in enumerate(pred_embeddings):
            distances, indices = nbrs.kneighbors([query_point])

            weights = 1.0 / (distances[0] + 1e-8)
            weights = weights / np.sum(weights)

            prediction = np.sum(weights * lib_targets[indices[0]])
            predictions.append(prediction)
            actuals.append(actual_targets[i])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        if np.std(predictions) == 0 or np.std(actuals) == 0:
            return 0.0

        correlation = np.corrcoef(predictions, actuals)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0


# ENHANCED SIMULATION FUNCTIONS (from original script)
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
                 coupling * x1[i - 1] +
                 noise_level * np.random.normal())

        strength_12 = current_strength * 0.9
        x2[i] = (strength_12 * x1[i - delay] +
                 coupling * x2[i - 1] +
                 noise_level * np.random.normal())

        strength_23 = current_strength * 0.7
        x3[i] = (strength_23 * x2[i - delay] +
                 coupling * x3[i - 1] +
                 noise_level * 1.1 * np.random.normal())

        strength_34 = current_strength * 0.5
        x4[i] = (strength_34 * x3[i - delay] +
                 coupling * x4[i - 1] +
                 noise_level * 1.2 * np.random.normal())

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
                 coupling * x1[i - 1] +
                 noise_level * np.random.normal())

        current_strong = strong_strength * strong_var
        x2[i] = (current_strong * x1[i - delay] +
                 coupling * x2[i - 1] +
                 noise_level * 0.9 * np.random.normal())

        current_weak = weak_strength * weak_var
        if np.sin(0.03 * i) > 0.3:
            x3[i] = (current_weak * x1[i - delay] +
                     coupling * x3[i - 1] +
                     noise_level * 1.3 * np.random.normal())
        else:
            x3[i] = (current_weak * 0.2 * x1[i - delay] +
                     coupling * x3[i - 1] +
                     noise_level * 1.5 * np.random.normal())

        x4[i] = (0.6 * np.cos(0.11 * i) + 0.4 * np.sin(0.07 * i) +
                 coupling * 0.9 * x4[i - 1] +
                 noise_level * np.random.normal())

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
                 coupling * x1[i - 1] +
                 noise_level * np.random.normal())

        x4[i] = (0.7 * np.cos(0.12 * i) + 0.3 * np.sin(0.08 * i) +
                 coupling * 0.85 * x4[i - 1] +
                 noise_level * np.random.normal())

        strong_influence = strong_strength * strong_var * x1[i - delay]
        weak_influence = weak_strength * weak_var2 * x4[i - delay]
        interference = 0.02 * strong_influence * weak_influence

        x2[i] = (strong_influence + weak_influence + interference +
                 coupling * x2[i - 1] +
                 noise_level * 1.1 * np.random.normal())

        current_weak1 = weak_strength * weak_var1
        if np.cos(0.04 * i) > 0.2:
            x3[i] = (current_weak1 * x1[i - delay] +
                     coupling * x3[i - 1] +
                     noise_level * 1.4 * np.random.normal())
        else:
            x3[i] = (coupling * x3[i - 1] +
                     noise_level * 1.6 * np.random.normal())

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


# ANALYSIS FUNCTIONS
def run_lstm_ccm_analysis(data_dict, var1, var2, embedding_dim=3, sequence_length=10,
                          lib_sizes=None, n_bootstrap=20, epochs=50):
    """Run LSTM-based CCM analysis between two variables"""
    if lib_sizes is None:
        lib_sizes = [80, 120, 160, 200]

    print(f"  Running LSTM-CCM analysis: {var1} ↔ {var2}")

    lstm_ccm = LSTMCCM(embedding_dim=embedding_dim,
                       sequence_length=sequence_length,
                       n_neighbors=min(5, max(3, embedding_dim)))

    lstm_ccm.fit_embedders(data_dict[var1], data_dict[var2], epochs=epochs, verbose=0)

    result = lstm_ccm.cross_map(data_dict[var1], data_dict[var2],
                                lib_sizes=lib_sizes, n_bootstrap=n_bootstrap)

    return result


def analyze_all_pairs_lstm(data_dict, embedding_dim=3, sequence_length=10,
                           lib_sizes=None, epochs=40):
    """LSTM-based analysis of all pairwise relationships"""
    if lib_sizes is None:
        lib_sizes = [80, 120, 160, 200]

    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        try:
            result = run_lstm_ccm_analysis(
                data_dict, var1, var2,
                embedding_dim=embedding_dim,
                sequence_length=sequence_length,
                lib_sizes=lib_sizes,
                epochs=epochs
            )
            results[f"{var1}_{var2}"] = result
        except Exception as e:
            print(f"  Error analyzing {var1} ↔ {var2}: {e}")
            results[f"{var1}_{var2}"] = None

    return results


def lstm_embedding_sensitivity_analysis(data_dict, var1, var2,
                                        embedding_dims=None, sequence_lengths=None,
                                        epochs=25):
    """Sensitivity analysis for LSTM embedding parameters"""
    if embedding_dims is None:
        embedding_dims = range(2, 8)
    if sequence_lengths is None:
        sequence_lengths = range(5, 16)

    results_matrix_xy = np.full((len(embedding_dims), len(sequence_lengths)), np.nan)
    results_matrix_yx = np.full((len(embedding_dims), len(sequence_lengths)), np.nan)

    print(f"  LSTM embedding sensitivity analysis for {var1} ↔ {var2}...")

    for i, embedding_dim in enumerate(embedding_dims):
        for j, sequence_length in enumerate(sequence_lengths):
            try:
                min_points = sequence_length + 100
                if len(data_dict[var1]) >= min_points:

                    result = run_lstm_ccm_analysis(
                        data_dict, var1, var2,
                        embedding_dim=embedding_dim,
                        sequence_length=sequence_length,
                        lib_sizes=[150],
                        n_bootstrap=10,
                        epochs=epochs
                    )

                    if result is not None and len(result) > 0:
                        final_row = result.iloc[-1]
                        results_matrix_xy[i, j] = final_row['X:Y']
                        results_matrix_yx[i, j] = final_row['Y:X']

            except Exception as e:
                print(f"    Error at embedding_dim={embedding_dim}, seq_len={sequence_length}: {str(e)[:40]}")
                results_matrix_xy[i, j] = np.nan
                results_matrix_yx[i, j] = np.nan

    return results_matrix_xy, results_matrix_yx, list(embedding_dims), list(sequence_lengths)


# VISUALIZATION FUNCTIONS
def visualize_time_series(data_dict, title, save_path=None):
    """Visualize the four time series"""
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
    """Create a causality matrix plot"""
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

                    if var1 < var2:
                        causality_matrix[i, j] = row['X:Y']
                    else:
                        causality_matrix[i, j] = row['Y:X']

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


def plot_lstm_sensitivity_heatmap(results_xy, results_yx, embedding_dims, sequence_lengths,
                                  var1, var2, title, save_path=None):
    """Plot heatmap for LSTM embedding parameter sensitivity"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin, vmax = -0.2, 0.7

    # Plot X→Y direction
    im1 = ax1.imshow(results_xy, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(f'{var1} → {var2}')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Embedding Dimension')
    ax1.set_xticks(range(len(sequence_lengths)))
    ax1.set_xticklabels(sequence_lengths)
    ax1.set_yticks(range(len(embedding_dims)))
    ax1.set_yticklabels(embedding_dims)

    # Add text annotations
    for i in range(len(embedding_dims)):
        for j in range(len(sequence_lengths)):
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
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Embedding Dimension')
    ax2.set_xticks(range(len(sequence_lengths)))
    ax2.set_xticklabels(sequence_lengths)
    ax2.set_yticks(range(len(embedding_dims)))
    ax2.set_yticklabels(embedding_dims)

    # Add text annotations
    for i in range(len(embedding_dims)):
        for j in range(len(sequence_lengths)):
            if not np.isnan(results_yx[i, j]):
                value = results_yx[i, j]
                color = 'white' if abs(value) > 0.35 else 'black'
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')
            else:
                ax2.text(j, i, 'N/A', ha='center', va='center',
                         color='gray', fontsize=7)

    plt.colorbar(im2, ax=ax2, label='CCM ρ')

    plt.suptitle(f'LSTM Embedding Parameter Sensitivity: {title}\n{var1} ↔ {var2}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved LSTM sensitivity plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def create_convergence_plot(results, key, title, save_path=None):
    """Create a CCM convergence plot for a specific pair of variables"""
    if key not in results or results[key] is None:
        print(f"No results for {key}")
        return

    var1, var2 = key.split('_')
    result = results[key]

    plt.figure(figsize=(12, 6))

    # Plot X predicts Y (var1 → var2)
    plt.plot(result['LibSize'], result['X:Y'], 'b-', marker='o', label=f'{var1} → {var2}')

    # Plot Y predicts X (var2 → var1)
    plt.plot(result['LibSize'], result['Y:X'], 'r-', marker='s', label=f'{var2} → {var1}')

    plt.xlabel('Library Size')
    plt.ylabel('Cross Map Skill (ρ)')
    plt.title(f'LSTM-CCM Convergence: {title} - {var1} ↔ {var2}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def create_all_convergence_plots(results, title, output_dir):
    """Create convergence plots for all pairs in the results"""
    for key in results:
        if results[key] is not None:
            var1, var2 = key.split('_')
            save_path = os.path.join(output_dir, f"{title}_LSTM_convergence_{var1}_{var2}.png")
            create_convergence_plot(results, key, title, save_path)


def plot_phase_space_reconstruction(data_dict, title, embedding_dim=3, sequence_length=10, save_path=None):
    """Create phase space reconstruction plots using LSTM embeddings"""
    from mpl_toolkits.mplot3d import Axes3D

    variables = ['X1', 'X2', 'X3', 'X4']
    colors = ['blue', 'red', 'green', 'purple']

    fig = plt.figure(figsize=(20, 15))

    for i, (var, color) in enumerate(zip(variables, colors)):
        data = data_dict[var]

        # Create embedder
        if TENSORFLOW_AVAILABLE:
            embedder = LSTMEmbedder(embedding_dim, sequence_length)
        else:
            embedder = SimplifiedEmbedder(embedding_dim, sequence_length)

        # Fit and embed
        embedder.fit(data, epochs=30, verbose=0)
        embeddings = embedder.embed(data)

        if embeddings.shape[1] >= 2:
            # 2D Phase Space (top row)
            ax_2d = plt.subplot(2, 4, i + 1)
            x = embeddings[:, 0]
            y = embeddings[:, 1]

            scatter = ax_2d.scatter(x, y, c=range(len(x)), cmap='viridis',
                                    alpha=0.6, s=1, edgecolors='none')
            ax_2d.set_xlabel(f'{var} Embedding Dim 1')
            ax_2d.set_ylabel(f'{var} Embedding Dim 2')
            ax_2d.set_title(f'2D LSTM Embedding: {var}')
            ax_2d.grid(True, alpha=0.3)

            # 3D Phase Space (bottom row) if embedding_dim >= 3
            if embeddings.shape[1] >= 3:
                ax_3d = plt.subplot(2, 4, i + 5, projection='3d')
                z = embeddings[:, 2]

                ax_3d.scatter(x, y, z, c=range(len(x)), cmap='viridis',
                              alpha=0.6, s=1, edgecolors='none')
                ax_3d.set_xlabel(f'{var} Embedding Dim 1')
                ax_3d.set_ylabel(f'{var} Embedding Dim 2')
                ax_3d.set_zlabel(f'{var} Embedding Dim 3')
                ax_3d.set_title(f'3D LSTM Embedding: {var}')

    embedder_type = "LSTM" if TENSORFLOW_AVAILABLE else "PCA"
    plt.suptitle(f'{title}\n{embedder_type} Embedding Space (dim={embedding_dim}, seq_len={sequence_length})',
                 fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved phase space plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def demonstrate_lstm_embedding_analysis(data_dict, simulation_name, output_dir):
    """Demonstrate LSTM embedding parameter sensitivity"""
    print(f"\n{'=' * 60}")
    print(f"LSTM EMBEDDING PARAMETER ANALYSIS: {simulation_name}")
    print(f"{'=' * 60}")

    # Test key relationships with LSTM analysis
    test_pairs = [('X1', 'X2'), ('X1', 'X3')]

    for var1, var2 in test_pairs:
        print(f"\nLSTM analysis for {var1} ↔ {var2}...")

        # LSTM sensitivity analysis
        results_xy, results_yx, embedding_dims, sequence_lengths = lstm_embedding_sensitivity_analysis(
            data_dict, var1, var2,
            embedding_dims=range(2, 7),
            sequence_lengths=range(6, 13),
            epochs=20
        )

        # Create heatmap
        heatmap_path = os.path.join(output_dir,
                                    f"{simulation_name.lower().replace(' ', '_')}_LSTM_sensitivity_{var1}_{var2}.png")
        plot_lstm_sensitivity_heatmap(results_xy, results_yx, embedding_dims, sequence_lengths,
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


def print_summary(results, simulation_name, expected_relationships):
    """Print a summary of the LSTM-CCM results"""
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

            # Check X predicts Y (var1 → var2)
            x_predicts_y = row['X:Y']
            if x_predicts_y > 0.3:
                print(f"  {var1} → {var2}: ρ = {x_predicts_y:.3f}")
                strong_relationships.append((var_to_idx[var1], var_to_idx[var2], x_predicts_y))

            # Check Y predicts X (var2 → var1)
            y_predicts_x = row['Y:X']
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


def save_results_to_csv(results, filename):
    """Save LSTM-CCM results to CSV file"""
    data_rows = []

    for key, result in results.items():
        if result is not None:
            source, target = key.split('_')

            for _, row in result.iterrows():
                data_dict = {
                    'Source': source,
                    'Target': target,
                    'Direction_X_Y': f"{source} → {target}",
                    'Direction_Y_X': f"{target} → {source}",
                    'Library_Size': row['LibSize'],
                    'X_predicts_Y': row['X:Y'],
                    'Y_predicts_X': row['Y:X'],
                    'X_predicts_Y_std': row['X:Y_std'],
                    'Y_predicts_X_std': row['Y:X_std']
                }
                data_rows.append(data_dict)

    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Saved results to: {filename}")


def run_lstm_ccm_simulations():
    """
    MAIN FUNCTION: Run all simulations with LSTM-CCM analysis
    """
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = f"LSTM_CCM_causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "lstm_ccm_simulation_log.txt")

    # Capture ALL output to log file
    with LogCapture(log_file):
        embedder_type = "LSTM" if TENSORFLOW_AVAILABLE else "PCA"
        print(f"Running Three Causality Simulations with {embedder_type}-CCM Analysis")
        print("=" * 80)
        print(f"Output will be saved to: {output_dir}")
        print(f"Log file: {log_file}")
        print(f"Using {embedder_type} embeddings for nonlinear time series analysis")

        # LSTM-CCM Simulation 1: Linear Chain
        print(f"\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4) with {embedder_type}-CCM")
        print("-" * 60)
        data1 = generate_enhanced_simulation_1(n_points=1200, base_strength=0.12, delay=3,
                                               noise_level=0.35, coupling=0.65)

        ts_plot1 = os.path.join(output_dir, "lstm_sim1_timeseries.png")
        visualize_time_series(data1, f"{embedder_type}-CCM Simulation 1: Linear Chain", save_path=ts_plot1)

        # LSTM embedding analysis
        demonstrate_lstm_embedding_analysis(data1, f"{embedder_type} Simulation 1", output_dir)

        print(f"  Analyzing causality with {embedder_type}-CCM...")
        results1 = analyze_all_pairs_lstm(data1, embedding_dim=4, sequence_length=10, epochs=35)
        expected1 = [(0, 1, "Strong"), (1, 2, "Medium"), (2, 3, "Weak")]

        causality_plot1 = os.path.join(output_dir, "lstm_sim1_causality.png")
        create_causality_plot(results1, f"{embedder_type}-CCM Simulation 1: Linear Chain", expected1,
                              save_path=causality_plot1)

        # Create phase space reconstruction plots
        phase_plot1 = os.path.join(output_dir, "lstm_sim1_phase_space.png")
        plot_phase_space_reconstruction(data1, f"{embedder_type}-CCM Simulation 1: Linear Chain",
                                        embedding_dim=4, sequence_length=10, save_path=phase_plot1)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results1, f"{embedder_type}_Sim1", output_dir)

        print_summary(results1, f"{embedder_type}-CCM Simulation 1", expected1)

        # LSTM-CCM Simulation 2: Hub with Outsider
        print(f"\n2. Running Simulation 2: Hub + Outsider with {embedder_type}-CCM")
        print("-" * 60)
        data2 = generate_enhanced_simulation_2(n_points=1200, strong_strength=0.15,
                                               weak_strength=0.04, delay=3,
                                               noise_level=0.35, coupling=0.65)

        ts_plot2 = os.path.join(output_dir, "lstm_sim2_timeseries.png")
        visualize_time_series(data2, f"{embedder_type}-CCM Simulation 2: Hub + Outsider", save_path=ts_plot2)

        demonstrate_lstm_embedding_analysis(data2, f"{embedder_type} Simulation 2", output_dir)

        print(f"  Analyzing causality with {embedder_type}-CCM...")
        results2 = analyze_all_pairs_lstm(data2, embedding_dim=4, sequence_length=10, epochs=35)
        expected2 = [(0, 1, "Strong"), (0, 2, "Weak")]

        causality_plot2 = os.path.join(output_dir, "lstm_sim2_causality.png")
        create_causality_plot(results2, f"{embedder_type}-CCM Simulation 2: Hub + Outsider", expected2,
                              save_path=causality_plot2)

        # Create phase space reconstruction plots
        phase_plot2 = os.path.join(output_dir, "lstm_sim2_phase_space.png")
        plot_phase_space_reconstruction(data2, f"{embedder_type}-CCM Simulation 2: Hub + Outsider",
                                        embedding_dim=4, sequence_length=10, save_path=phase_plot2)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results2, f"{embedder_type}_Sim2", output_dir)

        print_summary(results2, f"{embedder_type}-CCM Simulation 2", expected2)

        # LSTM-CCM Simulation 3: Complex Network
        print(f"\n3. Running Simulation 3: Complex Network with {embedder_type}-CCM")
        print("-" * 60)
        data3 = generate_enhanced_simulation_3(n_points=1200, strong_strength=0.14,
                                               weak_strength=0.05, delay=3,
                                               noise_level=0.35, coupling=0.65)

        ts_plot3 = os.path.join(output_dir, "lstm_sim3_timeseries.png")
        visualize_time_series(data3, f"{embedder_type}-CCM Simulation 3: Complex Network", save_path=ts_plot3)

        demonstrate_lstm_embedding_analysis(data3, f"{embedder_type} Simulation 3", output_dir)

        print(f"  Analyzing causality with {embedder_type}-CCM...")
        results3 = analyze_all_pairs_lstm(data3, embedding_dim=4, sequence_length=10, epochs=35)
        expected3 = [(0, 1, "Strong"), (0, 2, "Weak"), (3, 1, "Weak")]

        causality_plot3 = os.path.join(output_dir, "lstm_sim3_causality.png")
        create_causality_plot(results3, f"{embedder_type}-CCM Simulation 3: Complex Network", expected3,
                              save_path=causality_plot3)

        # Create phase space reconstruction plots
        phase_plot3 = os.path.join(output_dir, "lstm_sim3_phase_space.png")
        plot_phase_space_reconstruction(data3, f"{embedder_type}-CCM Simulation 3: Complex Network",
                                        embedding_dim=4, sequence_length=10, save_path=phase_plot3)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results3, f"{embedder_type}_Sim3", output_dir)

        print_summary(results3, f"{embedder_type}-CCM Simulation 3", expected3)

        # Save results
        save_results_to_csv(results1, os.path.join(output_dir, "lstm_sim1_results.csv"))
        save_results_to_csv(results2, os.path.join(output_dir, "lstm_sim2_results.csv"))
        save_results_to_csv(results3, os.path.join(output_dir, "lstm_sim3_results.csv"))

        # Final Summary
        print("\n" + "=" * 80)
        print(f"{embedder_type}-CCM ANALYSIS SUMMARY")
        print("=" * 80)
        print("Key Features of LSTM-CCM:")
        if TENSORFLOW_AVAILABLE:
            print("✓ LSTM neural network embeddings")
            print("✓ Nonlinear time series reconstruction")
            print("✓ Automatic feature learning")
            print("✓ Adaptive to data characteristics")
        else:
            print("✓ PCA-based embeddings (TensorFlow fallback)")
            print("✓ Linear dimensionality reduction")
            print("✓ Principal component analysis")
        print("✓ Cross-mapping in learned embedding space")
        print("✓ Parameter sensitivity analysis")
        print("✓ Convergence analysis with library size")

        print(f"\nExpected {embedder_type}-CCM Results:")
        print("• Strong connections: ρ = 0.4-0.8 (variable across parameters)")
        print("• Medium connections: ρ = 0.2-0.5 (more variable)")
        print("• Weak connections: ρ = 0.1-0.3 (highly variable)")
        print("• No connection: ρ = -0.1-0.1 (around zero)")

        print(f"\nAll {embedder_type}-CCM outputs saved to: {output_dir}")
        print("Check the sensitivity heatmaps to see parameter dependency!")

    # Print final message to console
    print(f"\n{embedder_type}-CCM simulation completed! Results saved to: {output_dir}")
    print("Key advantages over traditional CCM:")
    if TENSORFLOW_AVAILABLE:
        print("✓ Learns optimal nonlinear embeddings")
        print("✓ Captures complex temporal dependencies")
        print("✓ Better noise robustness")
    else:
        print("✓ Uses PCA for dimensionality reduction")
        print("✓ Linear embedding approximation")
    print("✓ Embedding space visualization available")


def compare_embedding_methods():
    """Quick comparison of different embedding approaches"""
    print("Comparing embedding methods on test data...")

    # Generate test data with known causality
    n_points = 500
    np.random.seed(42)
    x = np.cumsum(np.random.randn(n_points)) * 0.1
    y = np.zeros(n_points)
    for i in range(3, n_points):
        y[i] = 0.7 * x[i - 3] + 0.3 * y[i - 1] + 0.1 * np.random.randn()

    test_data = {'X1': x, 'X2': y, 'X3': np.random.randn(n_points), 'X4': np.random.randn(n_points)}

    # Test LSTM-CCM
    try:
        result = run_lstm_ccm_analysis(test_data, 'X1', 'X2',
                                       embedding_dim=3, sequence_length=8,
                                       epochs=25, lib_sizes=[100, 150])
        print(f"\n{('LSTM' if TENSORFLOW_AVAILABLE else 'PCA')}-CCM Test Results:")
        print(result)

        final_row = result.iloc[-1]
        print(f"X1→X2: {final_row['X:Y']:.3f} (should be low)")
        print(f"X2→X1: {final_row['Y:X']:.3f} (should be high)")

    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    print("LSTM-CCM Causality Analysis Script")
    print("==================================")

    if TENSORFLOW_AVAILABLE:
        print("✓ TensorFlow available - using LSTM embeddings")
    else:
        print("ℹ TensorFlow not available - using PCA embeddings")

    print("\nOptions:")
    print("1. Run quick comparison test")
    print("2. Run full simulation suite")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        compare_embedding_methods()
    else:
        # Run the full simulation
        run_lstm_ccm_simulations()