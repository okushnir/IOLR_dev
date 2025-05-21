import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_causal_series(n_points, causality_structure="x_causes_y", strength=0.3, delay=3, noise_level=0.1,
                           coupling=0.7):
    """
    Generate time series with different causality structures.

    Parameters:
    -----------
    n_points : int
        Length of time series
    causality_structure : str
        Type of causality: "x_causes_y", "y_causes_x", "bidirectional", "no_causality"
    strength : float
        Strength of causal influence (0-1)
    delay : int
        Time delay for causal effect
    noise_level : float
        Amount of noise to add
    coupling : float
        For y_causes_x or bidirectional, the coupling strength

    Returns:
    --------
    x, y : arrays
        Two time series with specified causality structure
    """

    np.random.seed(42)  # For reproducible results

    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Initialize with random values
    x[0] = np.random.random()
    y[0] = np.random.random()

    if causality_structure == "x_causes_y":
        # X causes Y with specified delay and strength
        for i in range(1, n_points):
            x[i] = np.sin(0.1 * i) + noise_level * np.random.random()
            if i > delay:
                y[i] = strength * x[i - delay] + coupling * y[i - 1] + noise_level * np.random.random()
            else:
                y[i] = coupling * y[i - 1] + noise_level * np.random.random()

    elif causality_structure == "y_causes_x":
        # Y causes X with specified delay and strength
        for i in range(1, n_points):
            y[i] = np.sin(0.1 * i) + noise_level * np.random.random()
            if i > delay:
                x[i] = strength * y[i - delay] + coupling * x[i - 1] + noise_level * np.random.random()
            else:
                x[i] = coupling * x[i - 1] + noise_level * np.random.random()

    elif causality_structure == "bidirectional":
        # Both X and Y cause each other
        for i in range(1, n_points):
            if i > delay:
                x[i] = np.sin(0.1 * i) + strength * y[i - delay] + coupling * x[
                    i - 1] + noise_level * np.random.random()
                y[i] = np.cos(0.1 * i) + strength * x[i - delay] + coupling * y[
                    i - 1] + noise_level * np.random.random()
            else:
                x[i] = np.sin(0.1 * i) + coupling * x[i - 1] + noise_level * np.random.random()
                y[i] = np.cos(0.1 * i) + coupling * y[i - 1] + noise_level * np.random.random()

    elif causality_structure == "no_causality":
        # Independent time series with no causal relationship
        for i in range(1, n_points):
            x[i] = np.sin(0.1 * i) + coupling * x[i - 1] + noise_level * np.random.random()
            y[i] = np.cos(0.15 * i) + coupling * y[i - 1] + noise_level * np.random.random()

    return x, y


def generate_lorenz_system(x_causes_y=True, strength=0.3, n_points=1000):
    """
    Generate coupled Lorenz attractors with causality.
    """
    dt = 0.01

    # Lorenz parameters
    sigma1, rho1, beta1 = 10.0, 28.0, 8.0 / 3.0
    sigma2, rho2, beta2 = 10.0, 28.0, 8.0 / 3.0

    # Initial conditions
    x1, y1, z1 = 1.0, 1.0, 1.0
    x2, y2, z2 = 2.0, 1.0, 1.0

    X = np.zeros(n_points)
    Y = np.zeros(n_points)

    for i in range(n_points):
        X[i] = x1
        Y[i] = x2

        # Lorenz system 1
        dx1 = sigma1 * (y1 - x1) * dt
        dy1 = (x1 * (rho1 - z1) - y1) * dt
        dz1 = (x1 * y1 - beta1 * z1) * dt

        # Lorenz system 2 (with coupling)
        if x_causes_y:
            coupling_term = strength * (x1 - x2)
        else:
            coupling_term = 0

        dx2 = (sigma2 * (y2 - x2) + coupling_term) * dt
        dy2 = (x2 * (rho2 - z2) - y2) * dt
        dz2 = (x2 * y2 - beta2 * z2) * dt

        # Update
        x1 += dx1
        y1 += dy1
        z1 += dz1
        x2 += dx2
        y2 += dy2
        z2 += dz2

    return X, Y


def test_different_causality_structures():
    """
    Test CCM on different causality structures to verify detection.
    """
    import os
    import sys
    sys.path.append(os.getcwd())  # Add current directory to path

    from pyedm_ccm_fixed import run_ccm_simulations, plot_ccm_results

    structures = ["x_causes_y", "y_causes_x", "bidirectional", "no_causality"]

    for structure in structures:
        print(f"\n{'=' * 50}")
        print(f"Testing: {structure}")
        print(f"{'=' * 50}")

        # Generate data with specific causality structure
        x, y = generate_causal_series(
            n_points=1000,
            causality_structure=structure,
            strength=0.5,  # Strong causality
            delay=3,
            noise_level=0.1
        )

        # Run CCM analysis
        results = run_ccm_simulations(
            time_series_1=x,
            time_series_2=y,
            num_simulations=10,
            lib_sizes=[50, 100, 150, 200],
            verbose=True
        )

        if results:
            # Plot results
            plot_ccm_results(results, f"CCM Analysis: {structure}")

            # Print summary
            max_lib = max(results['lib_sizes'])
            if max_lib in results['X_predicts_Y']:
                xy_rho = results['X_predicts_Y'][max_lib]['mean']
                print(f"X → Y: ρ = {xy_rho:.3f}")
            if max_lib in results['Y_predicts_X']:
                yx_rho = results['Y_predicts_X'][max_lib]['mean']
                print(f"Y → X: ρ = {yx_rho:.3f}")


def visualize_generated_series():
    """
    Visualize different causality structures.
    """
    structures = ["x_causes_y", "y_causes_x", "bidirectional", "no_causality"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, structure in enumerate(structures):
        x, y = generate_causal_series(
            n_points=500,
            causality_structure=structure,
            strength=0.5,
            delay=3
        )

        axes[i].plot(x[:100], label='X', alpha=0.7)
        axes[i].plot(y[:100], label='Y', alpha=0.7)
        axes[i].set_title(f'{structure.replace("_", " ").title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example configurations for different scenarios
if __name__ == "__main__":
    print("Testing different causality configurations...")

    # Visualize different structures
    visualize_generated_series()

    # Test a specific configuration
    print("\nTesting X causes Y configuration...")
    x, y = generate_causal_series(
        n_points=100,
        causality_structure="x_causes_y",
        strength=0.7,  # Strong causality
        delay=5,  # 5-step delay
        noise_level=0.05,  # Low noise
        coupling=0.8  # Strong autocorrelation
    )

    # You can now use these x, y series with your CCM function
    print("Generated time series with X causing Y")
    print(f"X mean: {np.mean(x):.3f}, std: {np.std(x):.3f}")
    print(f"Y mean: {np.mean(y):.3f}, std: {np.std(y):.3f}")