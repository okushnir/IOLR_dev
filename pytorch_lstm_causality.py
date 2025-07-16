import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import warnings
from datetime import datetime
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    print("Using PyTorch")
except ImportError:
    print("Install PyTorch: pip install torch")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for causality analysis"""

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take last output
        out = self.fc(out)
        return out


def create_sequences(data, sequence_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def train_lstm_model(X_train, y_train, X_val, y_val, sequence_length,
                     hidden_size=50, num_epochs=100, learning_rate=0.001):
    """Train LSTM model"""
    model = LSTMModel(input_size=1, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X.unsqueeze(-1))
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X.unsqueeze(-1))
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def run_pytorch_lstm_causality(data_dict, var1, var2, sequence_length=20):
    """PyTorch LSTM causality analysis"""
    series1 = np.array(data_dict[var1])
    series2 = np.array(data_dict[var2])

    # Normalize
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    series1_scaled = scaler1.fit_transform(series1.reshape(-1, 1)).flatten()
    series2_scaled = scaler2.fit_transform(series2.reshape(-1, 1)).flatten()

    # Create sequences for cross-prediction (var1 -> var2)
    X1, y2 = create_sequences(series1_scaled, sequence_length)
    y2_target = series2_scaled[sequence_length:]

    min_len = min(len(X1), len(y2_target))
    X1 = X1[:min_len]
    y2_target = y2_target[:min_len]

    if len(X1) < 100:
        return {'causality_score': 0.0, 'cross_r2': 0.0, 'self_r2': 0.0}

    # Split data
    X1_train, X1_test, y2_train, y2_test = train_test_split(
        X1, y2_target, test_size=0.2, random_state=42)

    # Train cross-prediction model
    model_cross = train_lstm_model(X1_train, y2_train, X1_test, y2_test, sequence_length)

    # Predict
    model_cross.eval()
    with torch.no_grad():
        X1_test_tensor = torch.FloatTensor(X1_test).unsqueeze(-1)
        y2_pred_cross = model_cross(X1_test_tensor).squeeze().numpy()

    r2_cross = r2_score(y2_test, y2_pred_cross)

    # Self-prediction model (var2 -> var2)
    X2, y2_self = create_sequences(series2_scaled, sequence_length)

    min_len_self = min(len(X2), len(y2_self))
    X2 = X2[:min_len_self]
    y2_self = y2_self[:min_len_self]

    X2_train, X2_test, y2_self_train, y2_self_test = train_test_split(
        X2, y2_self, test_size=0.2, random_state=42)

    model_self = train_lstm_model(X2_train, y2_self_train, X2_test, y2_self_test, sequence_length)

    model_self.eval()
    with torch.no_grad():
        X2_test_tensor = torch.FloatTensor(X2_test).unsqueeze(-1)
        y2_pred_self = model_self(X2_test_tensor).squeeze().numpy()

    r2_self = r2_score(y2_self_test, y2_pred_self)

    # Calculate causality
    causality_strength = max(0, r2_cross - r2_self)
    causality_score = min(1.0, causality_strength * 5)

    return {
        'cross_r2': r2_cross,
        'self_r2': r2_self,
        'causality_strength': causality_strength,
        'causality_score': causality_score
    }


def analyze_all_pairs_pytorch(data_dict):
    """Analyze all pairs with PyTorch LSTM"""
    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        print(f"  Analyzing {var1} ↔ {var2} with PyTorch LSTM...")

        result_xy = run_pytorch_lstm_causality(data_dict, var1, var2)
        result_yx = run_pytorch_lstm_causality(data_dict, var2, var1)

        results[f"{var1}_{var2}"] = {
            f"{var1}_to_{var2}": result_xy,
            f"{var2}_to_{var1}": result_yx
        }

    return results


# Include simulation functions
def generate_enhanced_simulation_1(n_points=1000, base_strength=0.12, delay=3,
                                   noise_level=0.35, coupling=0.65, seed=42):
    """Linear chain simulation"""
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
    """Hub + outsider simulation"""
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


def visualize_time_series(data_dict, title, save_path=None):
    """Visualize time series"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    variables = ['X1', 'X2', 'X3', 'X4']
    colors = ['blue', 'red', 'green', 'purple']

    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i // 2, i % 2]
        ax.plot(data_dict[var][:300], color=color, alpha=0.8)
        ax.set_title(f'{var} Time Series')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Value')
        ax.set_xlabel('Time')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def create_causality_plot(results, title, save_path=None):
    """Create causality matrix"""
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
    plt.colorbar(im, label='PyTorch LSTM Causality Score')

    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nPyTorch LSTM Causality Matrix')

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


def print_summary(results, simulation_name):
    """Print summary"""
    print(f"\nPyTorch LSTM Summary - {simulation_name}")
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


def run_pytorch_simulations():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"PYTORCH_LSTM_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("PyTorch LSTM Causality Analysis")
    print("=" * 40)

    # Simulation 1
    print("\n1. Linear Chain")
    data1 = generate_enhanced_simulation_1(n_points=800)

    visualize_time_series(data1, "PyTorch LSTM Sim 1: Linear Chain",
                          os.path.join(output_dir, "pytorch_sim1_timeseries.png"))

    results1 = analyze_all_pairs_pytorch(data1)

    create_causality_plot(results1, "PyTorch LSTM Sim 1: Linear Chain",
                          os.path.join(output_dir, "pytorch_sim1_causality.png"))

    print_summary(results1, "Linear Chain")

    # Simulation 2
    print("\n2. Hub + Outsider")
    data2 = generate_enhanced_simulation_2(n_points=800)

    visualize_time_series(data2, "PyTorch LSTM Sim 2: Hub + Outsider",
                          os.path.join(output_dir, "pytorch_sim2_timeseries.png"))

    results2 = analyze_all_pairs_pytorch(data2)

    create_causality_plot(results2, "PyTorch LSTM Sim 2: Hub + Outsider",
                          os.path.join(output_dir, "pytorch_sim2_causality.png"))

    print_summary(results2, "Hub + Outsider")

    print(f"\nCompleted! Results in: {output_dir}")


if __name__ == "__main__":
    run_pytorch_simulations()