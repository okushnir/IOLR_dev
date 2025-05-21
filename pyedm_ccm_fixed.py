import numpy as np
import pandas as pd
import os
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

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

# Import pyEDM after setting environment variables
import sys

sys.tracebacklimit = 0  # Reduce traceback verbosity

try:
    from pyEDM import CCM
except ImportError as e:
    print(f"Error importing pyEDM: {e}")
    print("Please fix the NumPy compatibility issue first")
    exit(1)


# Move the worker function outside the main function to avoid pickling issues
def run_single_ccm_worker(params):
    """Worker function for individual CCM simulation - must be at module level for pickling"""
    sim_id, random_seed, data_dict, E, tau, prediction_steps, lib_sizes = params

    # Recreate the dataframe from dictionary
    data = pd.DataFrame(data_dict)

    np.random.seed(random_seed)

    try:
        # For pyEDM, libSizes can be a list or a string with specific format
        # Let's try different approaches

        results_xy = []
        results_yx = []

        # Run CCM for each library size individually
        for lib_size in lib_sizes:
            try:
                # CCM: X predicts Y (test if X causes Y)
                ccm_xy = CCM(dataFrame=data,
                             E=E,
                             Tp=prediction_steps,
                             columns="X",
                             target="Y",
                             libSizes=str(lib_size),  # Single library size
                             sample=100,
                             tau=tau,
                             seed=random_seed)
                results_xy.append(ccm_xy)

                # CCM: Y predicts X (test if Y causes X)
                ccm_yx = CCM(dataFrame=data,
                             E=E,
                             Tp=prediction_steps,
                             columns="Y",
                             target="X",
                             libSizes=str(lib_size),  # Single library size
                             sample=100,
                             tau=tau,
                             seed=random_seed)
                results_yx.append(ccm_yx)

            except Exception as e:
                print(f"Error with library size {lib_size} in simulation {sim_id}: {e}")
                continue

        # Combine results
        if results_xy and results_yx:
            combined_xy = pd.concat(results_xy, ignore_index=True)
            combined_yx = pd.concat(results_yx, ignore_index=True)

            return {
                'sim_id': sim_id,
                'ccm_xy': combined_xy,
                'ccm_yx': combined_yx
            }
        else:
            return None

    except Exception as e:
        print(f"Error in simulation {sim_id}: {e}")
        return None


def run_ccm_simulations(time_series_1, time_series_2, num_simulations=100,
                        lib_sizes=None, E=3, tau=1, prediction_steps=1,
                        num_threads=None, verbose=True):
    """
    Run multiple CCM simulations to test causality between two time series.

    Parameters:
    -----------
    time_series_1 : array-like
        First time series (potential cause)
    time_series_2 : array-like
        Second time series (potential effect)
    num_simulations : int
        Number of CCM simulations to run
    lib_sizes : list, optional
        List of library sizes to test. If None, uses default range
    E : int
        Embedding dimension
    tau : int
        Time delay
    prediction_steps : int
        Number of steps ahead to predict
    num_threads : int, optional
        Number of threads for parallel processing. If None, uses CPU count
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict : Dictionary containing simulation results and statistics
    """

    # Ensure NumPy arrays
    time_series_1 = np.asarray(time_series_1)
    time_series_2 = np.asarray(time_series_2)

    # Set default library sizes if not provided
    if lib_sizes is None:
        max_lib = min(len(time_series_1), len(time_series_2)) - 20
        lib_sizes = list(range(10, max_lib, 5))

    # Set default number of threads
    if num_threads is None:
        num_threads = min(mp.cpu_count(), 4)  # Limit to 4 to avoid issues

    # Prepare data as dictionary (can be pickled)
    data_dict = {
        'time': list(range(len(time_series_1))),
        'X': time_series_1.tolist(),
        'Y': time_series_2.tolist()
    }

    # Generate random seeds for simulations
    random_seeds = np.random.randint(0, 1000000, size=num_simulations)

    # Prepare parameters for worker function
    params = [(i, seed, data_dict, E, tau, prediction_steps, lib_sizes)
              for i, seed in enumerate(random_seeds)]

    if verbose:
        print(f"Running {num_simulations} CCM simulations with {num_threads} threads...")

    # Run simulations
    results = []
    if num_threads == 1:
        # Sequential execution if single thread
        for i, param in enumerate(params):
            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_simulations} simulations")
            result = run_single_ccm_worker(param)
            if result is not None:
                results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            # Submit all jobs
            futures = [executor.submit(run_single_ccm_worker, param) for param in params]

            # Collect results as they complete
            for i, future in enumerate(futures):
                if verbose and (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{num_simulations} simulations")

                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"Error in parallel execution: {e}")

    if verbose:
        print(f"Successfully completed {len(results)}/{num_simulations} simulations")

    # Aggregate results
    aggregate_results = aggregate_ccm_results(results, lib_sizes)

    return aggregate_results


def aggregate_ccm_results(results, lib_sizes):
    """
    Aggregate results from multiple CCM simulations.
    """
    if not results:
        return None

    # First, let's examine what columns are in the CCM results
    if results:
        sample_result = results[0]
        print("CCM result columns:", sample_result['ccm_xy'].columns.tolist())
        print("Sample CCM result:")
        print(sample_result['ccm_xy'].head())

    # Collect all CCM values for each direction and library size
    ccm_xy_values = {lib_size: [] for lib_size in lib_sizes}
    ccm_yx_values = {lib_size: [] for lib_size in lib_sizes}

    for result in results:
        ccm_xy = result['ccm_xy']
        ccm_yx = result['ccm_yx']

        # Check if the column exists and find the right correlation column
        if 'rho' in ccm_xy.columns:
            corr_col = 'rho'
        elif 'Rho' in ccm_xy.columns:
            corr_col = 'Rho'
        elif 'correlation' in ccm_xy.columns:
            corr_col = 'correlation'
        elif 'pearson' in ccm_xy.columns:
            corr_col = 'pearson'
        else:
            # If we can't find the correlation column, let's use the first numeric column
            numeric_cols = ccm_xy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:  # Skip LibSize which should be first
                corr_col = numeric_cols[1]
            else:
                print("Error: Cannot find correlation column in CCM results")
                return None

        for _, row in ccm_xy.iterrows():
            lib_size = int(row['LibSize']) if 'LibSize' in row else int(row['lib_size']) if 'lib_size' in row else None
            if lib_size is not None and lib_size in ccm_xy_values:
                ccm_xy_values[lib_size].append(row[corr_col])

        for _, row in ccm_yx.iterrows():
            lib_size = int(row['LibSize']) if 'LibSize' in row else int(row['lib_size']) if 'lib_size' in row else None
            if lib_size is not None and lib_size in ccm_yx_values:
                ccm_yx_values[lib_size].append(row[corr_col])

    # Calculate statistics
    xy_stats = {}
    yx_stats = {}

    for lib_size in lib_sizes:
        if ccm_xy_values[lib_size]:
            xy_values = np.array(ccm_xy_values[lib_size])
            xy_stats[lib_size] = {
                'mean': np.mean(xy_values),
                'std': np.std(xy_values),
                'median': np.median(xy_values),
                'ci_lower': np.percentile(xy_values, 2.5),
                'ci_upper': np.percentile(xy_values, 97.5),
                'values': xy_values
            }

        if ccm_yx_values[lib_size]:
            yx_values = np.array(ccm_yx_values[lib_size])
            yx_stats[lib_size] = {
                'mean': np.mean(yx_values),
                'std': np.std(yx_values),
                'median': np.median(yx_values),
                'ci_lower': np.percentile(yx_values, 2.5),
                'ci_upper': np.percentile(yx_values, 97.5),
                'values': yx_values
            }

    return {
        'lib_sizes': lib_sizes,
        'X_predicts_Y': xy_stats,
        'Y_predicts_X': yx_stats,
        'raw_results': results
    }


def plot_ccm_results(results, title="CCM Analysis Results"):
    """
    Plot the results of CCM simulations.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Cannot plot results.")
        return

    if not results:
        print("No results to plot")
        return

    lib_sizes = results['lib_sizes']
    xy_stats = results['X_predicts_Y']
    yx_stats = results['Y_predicts_X']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot X predicts Y
    xy_means = [xy_stats[lib]['mean'] for lib in lib_sizes if lib in xy_stats]
    xy_ci_lower = [xy_stats[lib]['ci_lower'] for lib in lib_sizes if lib in xy_stats]
    xy_ci_upper = [xy_stats[lib]['ci_upper'] for lib in lib_sizes if lib in xy_stats]
    valid_lib_sizes_xy = [lib for lib in lib_sizes if lib in xy_stats]

    if xy_means:
        ax1.plot(valid_lib_sizes_xy, xy_means, 'b-', linewidth=2, label='Mean ρ')
        ax1.fill_between(valid_lib_sizes_xy, xy_ci_lower, xy_ci_upper,
                         alpha=0.3, color='blue', label='95% CI')
    ax1.set_xlabel('Library Size')
    ax1.set_ylabel('Cross-map skill (ρ)')
    ax1.set_title('X causes Y (X xM Y)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Y predicts X
    yx_means = [yx_stats[lib]['mean'] for lib in lib_sizes if lib in yx_stats]
    yx_ci_lower = [yx_stats[lib]['ci_lower'] for lib in lib_sizes if lib in yx_stats]
    yx_ci_upper = [yx_stats[lib]['ci_upper'] for lib in lib_sizes if lib in yx_stats]
    valid_lib_sizes_yx = [lib for lib in lib_sizes if lib in yx_stats]

    if yx_means:
        ax2.plot(valid_lib_sizes_yx, yx_means, 'r-', linewidth=2, label='Mean ρ')
        ax2.fill_between(valid_lib_sizes_yx, yx_ci_lower, yx_ci_upper,
                         alpha=0.3, color='red', label='95% CI')
    ax2.set_xlabel('Library Size')
    ax2.set_ylabel('Cross-map skill (ρ)')
    ax2.set_title('Y causes X (Y xM X)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # First, let's test a simple CCM to see what columns it returns
    print("Testing CCM to understand output format...")

    # Generate simple test data
    np.random.seed(42)
    n_points = 100
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)

    test_data = pd.DataFrame({'X': x, 'Y': y})

    try:
        test_ccm = CCM(dataFrame=test_data, E=2, columns="X", target="Y",
                       libSizes="10", sample=5)
        print("CCM output columns:", test_ccm.columns.tolist())
        print("Sample CCM output:")
        print(test_ccm)
    except Exception as e:
        print(f"Error in test CCM: {e}")

    print("\nNow running full simulation...")

    # Generate example data (you can replace this with your own time series)
    np.random.seed(42)
    n_points = 1000

    # Create two coupled time series
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Initialize
    x[0] = np.random.random()
    y[0] = np.random.random()

    # Generate coupled system where X causes Y
    for i in range(1, n_points):
        x[i] = np.sin(0.1 * i) + 0.1 * np.random.random()
        y[i] = 0.3 * x[i - 3] + 0.7 * y[i - 1] + 0.05 * np.random.random()

    # Run CCM simulations
    results = run_ccm_simulations(
        time_series_1=x,
        time_series_2=y,
        num_simulations=4,  # Start with a small number for testing
        lib_sizes=[20, 40, 60, 80, 100],  # Smaller list for testing
        E=3,
        tau=1,
        num_threads=2,  # Reduce number of threads
        verbose=True
    )

    # Plot results
    if results:
        plot_ccm_results(results, "CCM Analysis: Testing X → Y Causality")

        # Print summary statistics
        print("\nSummary (at largest library size):")
        max_lib = max(results['lib_sizes'])
        if max_lib in results['X_predicts_Y']:
            print(
                f"X → Y: ρ = {results['X_predicts_Y'][max_lib]['mean']:.3f} ± {results['X_predicts_Y'][max_lib]['std']:.3f}")
        if max_lib in results['Y_predicts_X']:
            print(
                f"Y → X: ρ = {results['Y_predicts_X'][max_lib]['mean']:.3f} ± {results['Y_predicts_X'][max_lib]['std']:.3f}")