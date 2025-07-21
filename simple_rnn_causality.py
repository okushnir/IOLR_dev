import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def create_temporal_features(series, lags=10):
    """Create temporal features without deep learning"""
    features = []
    for i in range(lags, len(series)):
        # Lagged values
        lag_features = [series[i - j] for j in range(1, lags + 1)]
        # Moving averages
        ma_3 = np.mean(series[i - 3:i])
        ma_5 = np.mean(series[i - 5:i])
        # Differences
        diff_1 = series[i - 1] - series[i - 2] if i >= 2 else 0
        diff_2 = series[i - 2] - series[i - 3] if i >= 3 else 0

        features.append(lag_features + [ma_3, ma_5, diff_1, diff_2])

    return np.array(features)


def run_temporal_causality(data_dict, var1, var2, lags=15):
    """Enhanced temporal causality with stronger detection"""
    series1 = np.array(data_dict[var1])
    series2 = np.array(data_dict[var2])

    # Use different preprocessing for better signal detection
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    combined = np.column_stack([series1, series2])
    normalized = scaler.fit_transform(combined)
    s1_norm, s2_norm = normalized[:, 0], normalized[:, 1]

    # Enhanced feature creation
    def create_rich_features(source_series, target_series, lags):
        features = []
        for i in range(lags, len(source_series) - 1):
            # Source variable features
            source_lags = [source_series[i - j] for j in range(1, lags + 1)]
            source_ma = np.mean(source_series[i - 5:i])
            source_std = np.std(source_series[i - 5:i])

            # Target variable past (for baseline)
            target_lags = [target_series[i - j] for j in range(1, min(lags, 5) + 1)]
            target_ma = np.mean(target_series[i - 3:i])

            # Interaction features
            correlation = np.corrcoef(source_series[i - lags:i], target_series[i - lags:i])[0, 1]
            correlation = 0 if np.isnan(correlation) else correlation

            features.append(source_lags + target_lags + [source_ma, source_std, target_ma, correlation])

        return np.array(features)

    # Cross-prediction features (source + target history)
    X_cross = create_rich_features(s1_norm, s2_norm, lags)
    # Baseline features (only target history)
    X_baseline = []
    for i in range(lags, len(s2_norm) - 1):
        baseline_features = [s2_norm[i - j] for j in range(1, lags + 1)]
        baseline_features += [np.mean(s2_norm[i - 5:i]), np.std(s2_norm[i - 5:i])]
        X_baseline.append(baseline_features)
    X_baseline = np.array(X_baseline)

    y_target = s2_norm[lags:len(s2_norm) - 1]

    # Ensure same length
    min_len = min(len(X_cross), len(X_baseline), len(y_target))
    X_cross = X_cross[:min_len]
    X_baseline = X_baseline[:min_len]
    y_target = y_target[:min_len]

    if len(X_cross) < 50:
        return {'causality_score': 0.0}

    # Train models with cross-validation
    from sklearn.model_selection import cross_val_score

    # Cross-prediction model (with source info)
    rf_cross = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    cv_scores_cross = cross_val_score(rf_cross, X_cross, y_target, cv=5, scoring='r2')
    r2_cross = np.mean(cv_scores_cross)

    # Baseline model (target only)
    rf_baseline = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    cv_scores_baseline = cross_val_score(rf_baseline, X_baseline, y_target, cv=5, scoring='r2')
    r2_baseline = np.mean(cv_scores_baseline)

    # Causality calculation with enhanced scaling
    causality_strength = max(0, r2_cross - r2_baseline)

    # More aggressive scaling for detection
    if causality_strength > 0.01:
        causality_score = min(1.0, causality_strength * 50)
    else:
        causality_score = 0.0

    return {
        'cross_r2': r2_cross,
        'baseline_r2': r2_baseline,
        'causality_strength': causality_strength,
        'causality_score': causality_score
    }


def generate_simulation_1(n_points=1000, seed=42):
    """Enhanced linear chain with stronger signals"""
    np.random.seed(seed)

    x = np.zeros((4, n_points))

    # Stronger coupling parameters
    for i in range(5, n_points):
        x[0, i] = 0.8 * np.sin(0.08 * i) + 0.7 * x[0, i - 1] + 0.25 * np.random.normal()
        x[1, i] = 0.25 * x[0, i - 3] + 0.7 * x[1, i - 1] + 0.3 * np.random.normal()  # Stronger
        x[2, i] = 0.18 * x[1, i - 3] + 0.7 * x[2, i - 1] + 0.35 * np.random.normal()  # Stronger
        x[3, i] = 0.12 * x[2, i - 3] + 0.7 * x[3, i - 1] + 0.4 * np.random.normal()  # Stronger

    return {'X1': x[0], 'X2': x[1], 'X3': x[2], 'X4': x[3]}


def generate_simulation_2(n_points=1000, seed=42):
    """Enhanced hub with stronger signals"""
    np.random.seed(seed)

    x = np.zeros((4, n_points))

    for i in range(5, n_points):
        x[0, i] = 0.8 * np.sin(0.09 * i) + 0.7 * x[0, i - 1] + 0.25 * np.random.normal()
        x[1, i] = 0.3 * x[0, i - 3] + 0.7 * x[1, i - 1] + 0.28 * np.random.normal()  # Strong

        # Moderate connection (not too weak)
        weak_strength = 0.15 if np.sin(0.03 * i) > 0.2 else 0.08
        x[2, i] = weak_strength * x[0, i - 3] + 0.7 * x[2, i - 1] + 0.35 * np.random.normal()

        x[3, i] = 0.6 * np.cos(0.11 * i) + 0.7 * x[3, i - 1] + 0.3 * np.random.normal()

    return {'X1': x[0], 'X2': x[1], 'X3': x[2], 'X4': x[3]}


def test_individual_causality(data_dict, var1, var2):
    """Debug individual relationships"""
    print(f"\nTesting {var1} → {var2}:")

    # Basic correlation check
    corr = np.corrcoef(data_dict[var1], data_dict[var2])[0, 1]
    print(f"  Raw correlation: {corr:.3f}")

    # Lagged correlation
    best_lag_corr = 0
    for lag in range(1, 8):
        if len(data_dict[var1]) > lag:
            lag_corr = np.corrcoef(data_dict[var1][:-lag], data_dict[var2][lag:])[0, 1]
            if abs(lag_corr) > abs(best_lag_corr):
                best_lag_corr = lag_corr
    print(f"  Best lagged correlation: {best_lag_corr:.3f}")

    # Causality test
    result = run_temporal_causality(data_dict, var1, var2)
    print(f"  Causality score: {result['causality_score']:.3f}")
    print(f"  R² improvement: {result['causality_strength']:.4f}")

    return result


def create_convergence_plot(results, var1, var2, title, save_path=None):
    """Create convergence-style plot showing R² comparison"""
    key = f"{var1}_{var2}"
    if key not in results:
        return

    result = results[key]
    xy_result = result[f"{var1}_to_{var2}"]
    yx_result = result[f"{var2}_to_{var1}"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # R² comparison
    metrics = ['Cross R²', 'Baseline R²', 'Causality Score']
    xy_values = [xy_result['cross_r2'], xy_result['baseline_r2'], xy_result['causality_score']]
    yx_values = [yx_result['cross_r2'], yx_result['baseline_r2'], yx_result['causality_score']]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width / 2, xy_values, width, label=f'{var1} → {var2}', alpha=0.8)
    ax1.bar(x + width / 2, yx_values, width, label=f'{var2} → {var1}', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Performance: {var1} ↔ {var2}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Causality strength
    directions = [f'{var1}→{var2}', f'{var2}→{var1}']
    strengths = [xy_result['causality_strength'], yx_result['causality_strength']]

    bars = ax2.bar(directions, strengths, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Causality Strength')
    ax2.set_title('Causality Comparison')
    ax2.grid(True, alpha=0.3)

    for bar, value in zip(bars, strengths):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.suptitle(f'{title}: {var1} ↔ {var2}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_convergence_plots(results, title, output_dir):
    """Create convergence plots for all pairs"""
    for var1, var2 in combinations(['X1', 'X2', 'X3', 'X4'], 2):
        save_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_convergence_{var1}_{var2}.png")
        create_convergence_plot(results, var1, var2, title, save_path)


def save_results_to_csv(results, filename):
    """Save results to CSV"""
    data_rows = []
    for key, result in results.items():
        var1, var2 = key.split('_')
        xy_result = result[f"{var1}_to_{var2}"]
        yx_result = result[f"{var2}_to_{var1}"]

        data_rows.extend([
            {'Source': var1, 'Target': var2, 'Cross_R2': xy_result['cross_r2'],
             'Baseline_R2': xy_result['baseline_r2'], 'Causality_Score': xy_result['causality_score']},
            {'Source': var2, 'Target': var1, 'Cross_R2': yx_result['cross_r2'],
             'Baseline_R2': yx_result['baseline_r2'], 'Causality_Score': yx_result['causality_score']}
        ])

    pd.DataFrame(data_rows).to_csv(filename, index=False)


def run_simple_causality():
    """Enhanced main function with all plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"COMPLETE_causality_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Complete Temporal Causality Analysis")
    print("=" * 40)

    # Linear Chain
    print("\n1. Linear Chain")
    data1 = generate_simulation_1(n_points=1000)

    test_individual_causality(data1, 'X1', 'X2')
    test_individual_causality(data1, 'X2', 'X3')

    visualize_time_series(data1, "Enhanced Linear Chain",
                          os.path.join(output_dir, "sim1_timeseries.png"))

    results1 = analyze_all_pairs(data1)

    create_causality_plot(results1, "Enhanced Linear Chain",
                          os.path.join(output_dir, "sim1_causality.png"))

    create_all_convergence_plots(results1, "Linear Chain", output_dir)
    save_results_to_csv(results1, os.path.join(output_dir, "sim1_results.csv"))

    print_summary(results1, "Enhanced Linear Chain")

    # Hub Network
    print("\n2. Hub Network")
    data2 = generate_simulation_2(n_points=1000)

    test_individual_causality(data2, 'X1', 'X2')
    test_individual_causality(data2, 'X1', 'X3')

    visualize_time_series(data2, "Enhanced Hub Network",
                          os.path.join(output_dir, "sim2_timeseries.png"))

    results2 = analyze_all_pairs(data2)

    create_causality_plot(results2, "Enhanced Hub Network",
                          os.path.join(output_dir, "sim2_causality.png"))

    create_all_convergence_plots(results2, "Hub Network", output_dir)
    save_results_to_csv(results2, os.path.join(output_dir, "sim2_results.csv"))

    print_summary(results2, "Enhanced Hub Network")

    print(f"\nCompleted! Full results: {output_dir}")
    print("Generated: causality matrices, convergence plots, CSV files")


def analyze_all_pairs(data_dict):
    """Analyze all variable pairs"""
    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    for var1, var2 in combinations(variables, 2):
        print(f"  {var1} ↔ {var2}")

        result_xy = run_temporal_causality(data_dict, var1, var2)
        result_yx = run_temporal_causality(data_dict, var2, var1)

        results[f"{var1}_{var2}"] = {
            f"{var1}_to_{var2}": result_xy,
            f"{var2}_to_{var1}": result_yx
        }

    return results


# Simulation functions
def generate_simulation_1(n_points=1000, seed=42):
    """Linear chain: 1 → 2 → 3 → 4"""
    np.random.seed(seed)

    x = np.zeros((4, n_points))

    for i in range(4, n_points):
        x[0, i] = 0.7 * np.sin(0.08 * i) + 0.6 * x[0, i - 1] + 0.3 * np.random.normal()
        x[1, i] = 0.12 * x[0, i - 3] + 0.6 * x[1, i - 1] + 0.35 * np.random.normal()
        x[2, i] = 0.08 * x[1, i - 3] + 0.6 * x[2, i - 1] + 0.4 * np.random.normal()
        x[3, i] = 0.05 * x[2, i - 3] + 0.6 * x[3, i - 1] + 0.45 * np.random.normal()

    return {'X1': x[0], 'X2': x[1], 'X3': x[2], 'X4': x[3]}


def generate_simulation_2(n_points=1000, seed=42):
    """Hub: 1 → 2 (strong), 1 → 3 (weak), 4 independent"""
    np.random.seed(seed)

    x = np.zeros((4, n_points))

    for i in range(4, n_points):
        x[0, i] = 0.8 * np.sin(0.09 * i) + 0.65 * x[0, i - 1] + 0.3 * np.random.normal()
        x[1, i] = 0.15 * x[0, i - 3] + 0.65 * x[1, i - 1] + 0.32 * np.random.normal()

        # Weak intermittent connection
        weak_strength = 0.04 if np.sin(0.03 * i) > 0.3 else 0.01
        x[2, i] = weak_strength * x[0, i - 3] + 0.65 * x[2, i - 1] + 0.4 * np.random.normal()

        x[3, i] = 0.6 * np.cos(0.11 * i) + 0.6 * x[3, i - 1] + 0.35 * np.random.normal()

    return {'X1': x[0], 'X2': x[1], 'X3': x[2], 'X4': x[3]}


def visualize_time_series(data_dict, title, save_path=None):
    """Plot time series"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    variables = ['X1', 'X2', 'X3', 'X4']
    colors = ['blue', 'red', 'green', 'purple']

    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i // 2, i % 2]
        ax.plot(data_dict[var][:200], color=color, alpha=0.8)
        ax.set_title(f'{var} Time Series')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def create_causality_plot(results, title, save_path=None):
    """Create causality matrix"""
    variables = ['X1', 'X2', 'X3', 'X4']
    matrix = np.zeros((4, 4))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results:
                    result = results[key]
                    if var1 < var2:
                        matrix[i, j] = result[f"{var1}_to_{var2}"]['causality_score']
                        matrix[j, i] = result[f"{var2}_to_{var1}"]['causality_score']

    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(im, label='Causality Score')

    plt.xticks(range(4), variables)
    plt.yticks(range(4), variables)
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.title(f'{title}\nCausality Matrix')

    for i in range(4):
        for j in range(4):
            if i != j:
                color = 'white' if matrix[i, j] > 0.5 else 'black'
                plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                         color=color, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def print_summary(results, name):
    """Print results"""
    print(f"\n{name} Results:")
    for var1, var2 in combinations(['X1', 'X2', 'X3', 'X4'], 2):
        key = f"{var1}_{var2}"
        if key in results:
            xy = results[key][f"{var1}_to_{var2}"]['causality_score']
            yx = results[key][f"{var2}_to_{var1}"]['causality_score']

            if xy > 0.3:
                print(f"  {var1} → {var2}: {xy:.3f}")
            if yx > 0.3:
                print(f"  {var2} → {var1}: {yx:.3f}")


def run_simple_causality():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"SIMPLE_causality_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Simple Temporal Causality Analysis")
    print("=" * 40)

    # Linear Chain
    print("\n1. Linear Chain")
    data1 = generate_simulation_1(n_points=800)

    visualize_time_series(data1, "Linear Chain",
                          os.path.join(output_dir, "sim1_timeseries.png"))

    results1 = analyze_all_pairs(data1)

    create_causality_plot(results1, "Linear Chain",
                          os.path.join(output_dir, "sim1_causality.png"))

    print_summary(results1, "Linear Chain")

    # Hub Network
    print("\n2. Hub Network")
    data2 = generate_simulation_2(n_points=800)

    visualize_time_series(data2, "Hub Network",
                          os.path.join(output_dir, "sim2_timeseries.png"))

    results2 = analyze_all_pairs(data2)

    create_causality_plot(results2, "Hub Network",
                          os.path.join(output_dir, "sim2_causality.png"))

    print_summary(results2, "Hub Network")

    print(f"\nCompleted! Results: {output_dir}")


if __name__ == "__main__":
    run_simple_causality()