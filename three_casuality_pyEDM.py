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


def generate_simulation_1(n_points=1000, strength=0.2, delay=3, noise_level=0.5, coupling=0.8, seed=42):
    """
    Simulation 1: Linear chain - 1 → 2 → 3 → 4
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(4):
        x1[i] = np.random.random()
        x2[i] = np.random.random()
        x3[i] = np.random.random()
        x4[i] = np.random.random()

    # Generate the linear chain
    for i in range(4, n_points):
        # X1 is the driving force
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X2 is influenced by X1
        x2[i] = strength * x1[i - delay] + coupling * x2[i - 1] + noise_level * np.random.random()

        # X3 is influenced by X2
        x3[i] = strength * x2[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()

        # X4 is influenced by X3
        x4[i] = strength * x3[i - delay] + coupling * x4[i - 1] + noise_level * np.random.random()

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def generate_simulation_2(n_points=1000, strong_strength=0.2, weak_strength=0.01, delay=3, noise_level=0.5,
                          coupling=0.8,
                          seed=42):
    """
    Simulation 2: 1 → 2 (strong), 1 → 3 (weak), 4 is independent
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(4):
        x1[i] = np.random.random()
        x2[i] = np.random.random()
        x3[i] = np.random.random()
        x4[i] = np.random.random()

    # Generate the hub structure
    for i in range(4, n_points):
        # X1 is the driving force
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X2 is strongly influenced by X1
        x2[i] = strong_strength * x1[i - delay] + coupling * x2[i - 1] + noise_level * np.random.random()

        # X3 is weakly influenced by X1
        x3[i] = weak_strength * x1[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()

        # X4 is independent
        x4[i] = np.cos(0.12 * i) + coupling * x4[i - 1] + noise_level * np.random.random()

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def generate_simulation_3(n_points=1000, strong_strength=0.2, weak_strength=0.01, delay=3, noise_level=0.5,
                          coupling=0.8,
                          seed=42):
    """
    Simulation 3: 1 → 2 (strong), 1 → 3 (weak), 4 → 2 (weak)
    """
    np.random.seed(seed)

    x1 = np.zeros(n_points)
    x2 = np.zeros(n_points)
    x3 = np.zeros(n_points)
    x4 = np.zeros(n_points)

    # Initial values
    for i in range(4):
        x1[i] = np.random.random()
        x2[i] = np.random.random()
        x3[i] = np.random.random()
        x4[i] = np.random.random()

    # Generate the network structure
    for i in range(4, n_points):
        # X1 is the driving force
        x1[i] = np.sin(0.1 * i) + coupling * x1[i - 1] + noise_level * np.random.random()

        # X4 is independent (needs to be generated before X2 since X4 influences X2)
        x4[i] = np.cos(0.12 * i) + coupling * x4[i - 1] + noise_level * np.random.random()

        # X2 is influenced by both X1 (strong) and X4 (weak)
        x2[i] = (strong_strength * x1[i - delay] +
                 weak_strength * x4[i - delay] +
                 coupling * x2[i - 1] +
                 noise_level * np.random.random())

        # X3 is weakly influenced by X1
        x3[i] = weak_strength * x1[i - delay] + coupling * x3[i - 1] + noise_level * np.random.random()

    return {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4}


def run_ccm_direct(data_dict, var1, var2, E=3, tau=1, lib_sizes=None):
    """
    Run Convergent Cross Mapping directly using pyEDM's CCM function.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with time series data
    var1, var2 : str
        Names of variables to analyze
    E : int
        Embedding dimension
    tau : int
        Time delay
    lib_sizes : list
        Library sizes for convergence

    Returns:
    --------
    dict
        Dictionary with CCM results
    """
    if lib_sizes is None:
        lib_sizes = "50 100 150 200 250"
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
                         sample=100,  # Number of random samples at each library size
                         tau=tau,
                         seed=42)

        # Print the column names to debug
        print(f"  CCM result columns: {ccm_result.columns.tolist()}")
        print(f"  Sample CCM result:")
        print(f"  {ccm_result.head()}")

        # Rename columns for consistency if needed
        column_map = {}
        for col in ccm_result.columns:
            if col == 'LibSize':
                continue
            elif ':' in col:
                # Keep the original column name with ':'
                continue
            elif var1 in col and var2 in col and '->' in col:
                column_map[col] = 'X:Y'  # Rename to our expected format
            elif var2 in col and var1 in col and '->' in col:
                column_map[col] = 'Y:X'  # Rename to our expected format

        # Apply column renaming if needed
        if column_map:
            ccm_result = ccm_result.rename(columns=column_map)
            print(f"  Renamed columns: {column_map}")
            print(f"  New columns: {ccm_result.columns.tolist()}")

        # If we still don't have X:Y and Y:X columns, try to determine which are which
        if 'X:Y' not in ccm_result.columns or 'Y:X' not in ccm_result.columns:
            numeric_cols = [col for col in ccm_result.columns if col != 'LibSize']
            if len(numeric_cols) >= 2:
                ccm_result = ccm_result.rename(columns={
                    numeric_cols[0]: 'X:Y',
                    numeric_cols[1]: 'Y:X'
                })
                print(f"  Assigned columns: {numeric_cols[0]} → X:Y, {numeric_cols[1]} → Y:X")

        return ccm_result
    except Exception as e:
        print(f"  CCM error for {var1} ↔ {var2}: {e}")
        return None


def analyze_all_pairs(data_dict, E=3, tau=1, lib_sizes=None):
    """Analyze all pairwise relationships using CCM directly."""
    if lib_sizes is None:
        lib_sizes = [50, 100, 150, 200, 250]

    variables = ['X1', 'X2', 'X3', 'X4']
    results = {}

    # Analyze all pairwise relationships
    for var1, var2 in combinations(variables, 2):
        print(f"  Analyzing {var1} ↔ {var2}...")

        result = run_ccm_direct(data_dict, var1, var2, E, tau, lib_sizes)

        if result is not None:
            print(f"  CCM result columns: {result.columns.tolist()}")
            print(f"  Sample CCM result:")
            print(f"  {result.head()}")

            results[f"{var1}_{var2}"] = result

    return results


def create_causality_plot(results, title, expected_relationships=None, save_path=None):
    """Create a causality matrix plot."""
    variables = ['X1', 'X2', 'X3', 'X4']
    n_vars = len(variables)

    # Create matrix for causality
    causality_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                key = f"{var1}_{var2}" if f"{var1}_{var2}" in results else f"{var2}_{var1}"
                if key in results and results[key] is not None:
                    result = results[key]

                    # Get the maximum library size
                    max_lib = result['LibSize'].max()

                    # Get the row with the max library size
                    row = result[result['LibSize'] == max_lib].iloc[0]

                    # Determine which columns to use for causality values
                    x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
                    y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

                    # If X:Y and Y:X aren't available, try to find other numeric columns
                    if x_to_y_col is None or y_to_x_col is None:
                        numeric_cols = [col for col in result.columns if col != 'LibSize']
                        if len(numeric_cols) >= 2:
                            x_to_y_col = numeric_cols[0]
                            y_to_x_col = numeric_cols[1]

                    if var1 < var2:  # Original order
                        if x_to_y_col:
                            causality_matrix[i, j] = row[x_to_y_col]  # var1 predicts var2
                    else:  # Swapped order
                        if y_to_x_col:
                            causality_matrix[i, j] = row[y_to_x_col]  # var2 predicts var1

    # Plot the causality matrix
    plt.figure(figsize=(10, 8))

    # Create heatmap
    im = plt.imshow(causality_matrix, cmap='RdBu_r', vmin=-0.5, vmax=1)
    plt.colorbar(im, label='CCM ρ')

    # Set labels
    plt.xticks(range(n_vars), variables)
    plt.yticks(range(n_vars), variables)
    plt.xlabel('Target Variable')
    plt.ylabel('Source Variable')
    plt.title(f'{title}\nCausality Matrix (Row → Column)')

    # Add text annotations
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                color = 'white' if abs(causality_matrix[i, j]) > 0.5 else 'black'
                plt.text(j, i, f'{causality_matrix[i, j]:.2f}',
                         ha='center', va='center', color=color, fontweight='bold')

    # Highlight expected relationships if provided
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


def analyze_phase_space_patterns(data_dict, simulation_name, E=3, tau=1):
    """Analyze and provide diagnostic text for phase space patterns."""
    print(f"\n{'=' * 60}")
    print(f"PHASE SPACE ANALYSIS: {simulation_name}")
    print(f"{'=' * 60}")

    variables = ['X1', 'X2', 'X3', 'X4']
    patterns = {}

    for var in variables:
        data = data_dict[var]
        n_points = len(data) - (E - 1) * tau

        if n_points > 100:
            # Create 2D embedding
            x = data[:-tau][:n_points - tau]
            y = data[tau:][:n_points - tau]

            # Calculate pattern characteristics
            pattern_analysis = analyze_attractor_shape(x, y, var)
            patterns[var] = pattern_analysis

            print(f"\n{var} Analysis:")
            print(f"  Shape: {pattern_analysis['shape_description']}")
            print(f"  Regularity: {pattern_analysis['regularity_description']}")
            print(f"  Noise Level: {pattern_analysis['noise_description']}")
            print(f"  Interpretation: {pattern_analysis['interpretation']}")

    # Provide simulation-specific diagnostics
    print(f"\n{'=' * 40}")
    print("EXPECTED vs OBSERVED PATTERNS:")
    print(f"{'=' * 40}")

    if "Simulation 1" in simulation_name:
        analyze_simulation1_patterns(patterns)
    elif "Simulation 2" in simulation_name:
        analyze_simulation2_patterns(patterns)
    elif "Simulation 3" in simulation_name:
        analyze_simulation3_patterns(patterns)

    return patterns


def analyze_attractor_shape(x, y, var_name):
    """Analyze the shape and characteristics of a 2D attractor."""
    import numpy as np
    from scipy import stats

    # Calculate basic statistics
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)

    # Correlation between x and y coordinates
    correlation = np.corrcoef(x, y)[0, 1]

    # Calculate "thickness" of attractor (measure of noise)
    # Use distance from points to their best-fit ellipse/line
    if abs(correlation) > 0.7:  # Linear-ish pattern
        # Fit line and calculate residuals
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        predicted_y = slope * x + intercept
        residuals = y - predicted_y
        thickness = np.std(residuals)
        shape_type = "linear"
    else:
        # For non-linear patterns, use distance from centroid
        center_x, center_y = np.mean(x), np.mean(y)
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        thickness = np.std(distances) / np.mean(distances)
        shape_type = "curved"

    # Determine regularity (how predictable the pattern is)
    # Use autocorrelation of the trajectory
    trajectory_x = np.diff(x)
    trajectory_y = np.diff(y)
    trajectory_angles = np.arctan2(trajectory_y, trajectory_x)

    # Calculate how consistent the trajectory direction is
    angle_consistency = 1 - np.std(np.diff(trajectory_angles)) / np.pi

    # Classify patterns
    analysis = {
        'correlation': correlation,
        'thickness': thickness,
        'angle_consistency': angle_consistency,
        'x_range': x_range,
        'y_range': y_range,
        'shape_type': shape_type
    }

    # Generate descriptions
    analysis['shape_description'] = describe_shape(analysis)
    analysis['regularity_description'] = describe_regularity(analysis)
    analysis['noise_description'] = describe_noise(analysis)
    analysis['interpretation'] = interpret_pattern(analysis, var_name)

    return analysis


def describe_shape(analysis):
    """Describe the overall shape of the attractor."""
    corr = abs(analysis['correlation'])

    if corr > 0.9:
        return "Strong linear relationship"
    elif corr > 0.7:
        return "Moderate linear relationship"
    elif corr > 0.3:
        return "Weak linear relationship"
    elif analysis['shape_type'] == "curved" and analysis['angle_consistency'] > 0.5:
        return "Curved/elliptical pattern"
    else:
        return "Scattered/random pattern"


def describe_regularity(analysis):
    """Describe how regular/predictable the pattern is."""
    consistency = analysis['angle_consistency']

    if consistency > 0.8:
        return "Highly regular (very predictable)"
    elif consistency > 0.6:
        return "Moderately regular (somewhat predictable)"
    elif consistency > 0.4:
        return "Somewhat irregular (low predictability)"
    else:
        return "Highly irregular (unpredictable)"


def describe_noise(analysis):
    """Describe the noise level in the pattern."""
    thickness = analysis['thickness']

    if analysis['shape_type'] == "linear":
        if thickness < 0.1:
            return "Very low noise (clean signal)"
        elif thickness < 0.3:
            return "Low noise (clear signal)"
        elif thickness < 0.6:
            return "Moderate noise (signal visible)"
        else:
            return "High noise (signal obscured)"
    else:  # curved patterns
        if thickness < 0.2:
            return "Very low noise (clean pattern)"
        elif thickness < 0.4:
            return "Low noise (clear pattern)"
        elif thickness < 0.7:
            return "Moderate noise (pattern visible)"
        else:
            return "High noise (pattern obscured)"


def interpret_pattern(analysis, var_name):
    """Provide interpretation of what the pattern means."""
    shape = analysis['shape_description']
    noise = analysis['noise_description']
    regularity = analysis['regularity_description']

    if "Strong linear" in shape and "low noise" in noise.lower():
        return f"{var_name} shows strong deterministic coupling (good signal transmission)"
    elif "linear" in shape.lower() and "moderate" in noise.lower():
        return f"{var_name} shows causal influence with added noise (typical transmission)"
    elif "Curved" in shape and "regular" in regularity.lower():
        return f"{var_name} shows periodic/oscillatory behavior (driven by sin/cos)"
    elif "Scattered" in shape:
        return f"{var_name} shows weak coupling or independence (high noise/randomness)"
    else:
        return f"{var_name} shows complex dynamics (multiple influences or nonlinear effects)"


def analyze_simulation1_patterns(patterns):
    """Analyze patterns specific to Simulation 1 (Linear Chain)."""
    print("\nExpected for Linear Chain (X1→X2→X3→X4):")
    print("✓ X1: Clean curved/elliptical (sin wave)")
    print("✓ X2: Similar to X1 but slightly noisier")
    print("✓ X3: More degraded version of X1 pattern")
    print("✓ X4: Most degraded/noisy version")

    print(f"\nActual Results:")

    # Check if X1 is the cleanest
    x1_noise = "low noise" in patterns['X1']['noise_description'].lower()
    print(f"• X1 cleanest pattern: {'✓ YES' if x1_noise else '✗ NO - check parameters'}")

    # Check progression of degradation
    shapes = [patterns[var]['shape_description'] for var in ['X1', 'X2', 'X3', 'X4']]
    similar_shapes = sum(1 for i in range(3) if "linear" in shapes[i].lower() and "linear" in shapes[i + 1].lower())
    print(f"• Progressive degradation: {'✓ YES' if similar_shapes >= 2 else '✗ NO - coupling may be too weak'}")

    # Check if patterns are related
    correlations = [abs(patterns[var]['correlation']) for var in ['X1', 'X2', 'X3', 'X4']]
    decreasing = all(correlations[i] >= correlations[i + 1] - 0.2 for i in range(3))
    print(f"• Causal chain visible: {'✓ YES' if decreasing else '✗ NO - check strength parameters'}")


def analyze_simulation2_patterns(patterns):
    """Analyze patterns specific to Simulation 2 (Hub + Outsider)."""
    print("\nExpected for Hub + Outsider (X1→X2 strong, X1→X3 weak, X4 independent):")
    print("✓ X1: Clean curved/elliptical (sin wave)")
    print("✓ X2: Very similar to X1 (strong connection)")
    print("✓ X3: Weakly similar to X1 (weak connection)")
    print("✓ X4: Different pattern (cos wave, independent)")

    print(f"\nActual Results:")

    # Check X1-X2 strong similarity
    x1_shape = patterns['X1']['shape_description']
    x2_shape = patterns['X2']['shape_description']
    strong_connection = ("linear" in x1_shape.lower() and "linear" in x2_shape.lower()) or \
                        ("curved" in x1_shape.lower() and "curved" in x2_shape.lower())
    print(f"• X1→X2 strong connection: {'✓ YES' if strong_connection else '✗ NO - check strong_strength'}")

    # Check X1-X3 weak similarity
    x3_noise = patterns['X3']['noise_description']
    weak_connection = "moderate" in x3_noise.lower() or "high" in x3_noise.lower()
    print(f"• X1→X3 weak connection: {'✓ YES' if weak_connection else '✗ NO - check weak_strength'}")

    # Check X4 independence
    x4_pattern = patterns['X4']['shape_description']
    independent = "scattered" in x4_pattern.lower() or "curved" in x4_pattern.lower()
    print(f"• X4 independence: {'✓ YES' if independent else '✗ NO - check independence'}")


def analyze_simulation3_patterns(patterns):
    """Analyze patterns specific to Simulation 3 (Complex Network)."""
    print("\nExpected for Complex Network (X1→X2 strong, X1→X3 weak, X4→X2 weak):")
    print("✓ X1: Clean curved/elliptical (sin wave)")
    print("✓ X2: Complex pattern (influenced by both X1 and X4)")
    print("✓ X3: Weakly similar to X1")
    print("✓ X4: Different clean pattern (cos wave)")

    print(f"\nActual Results:")

    # Check X2 complexity (should be most complex due to dual influence)
    x2_interpretation = patterns['X2']['interpretation']
    complex_x2 = "complex" in x2_interpretation.lower() or "multiple" in x2_interpretation.lower()
    print(f"• X2 shows complex pattern: {'✓ YES' if complex_x2 else '✗ MAYBE - dual influence may be weak'}")

    # Check X1 and X4 are different
    x1_corr = abs(patterns['X1']['correlation'])
    x4_corr = abs(patterns['X4']['correlation'])
    different_drivers = abs(x1_corr - x4_corr) > 0.3
    print(f"• X1 and X4 are different: {'✓ YES' if different_drivers else '✗ NO - check driving functions'}")

    # Check X3 weakness
    x3_noise = patterns['X3']['noise_description']
    weak_x3 = "moderate" in x3_noise.lower() or "high" in x3_noise.lower()
    print(f"• X3 shows weak influence: {'✓ YES' if weak_x3 else '✗ NO - check weak_strength'}")


def embedding_sensitivity_analysis(data_dict, var1, var2, E_range=None, tau_range=None):
    """Test CCM across different embedding parameters to show dependency."""
    if E_range is None:
        E_range = range(1, 7)
    if tau_range is None:
        tau_range = range(1, 6)

    results_matrix_xy = np.zeros((len(E_range), len(tau_range)))
    results_matrix_yx = np.zeros((len(E_range), len(tau_range)))

    print(f"  Testing embedding parameters for {var1} ↔ {var2}...")

    for i, E in enumerate(E_range):
        for j, tau in enumerate(tau_range):
            try:
                # Test if we have enough data points for this embedding
                n_points = len(data_dict[var1]) - (E - 1) * tau
                if n_points > 100:  # Minimum points needed
                    result = run_ccm_direct(data_dict, var1, var2, E=E, tau=tau, lib_sizes=[250])
                    if result is not None and len(result) > 0:
                        # Get final correlation values
                        final_row = result.iloc[-1]
                        if 'X:Y' in result.columns:
                            results_matrix_xy[i, j] = final_row['X:Y']
                        if 'Y:X' in result.columns:
                            results_matrix_yx[i, j] = final_row['Y:X']
                    else:
                        results_matrix_xy[i, j] = np.nan
                        results_matrix_yx[i, j] = np.nan
                else:
                    results_matrix_xy[i, j] = np.nan
                    results_matrix_yx[i, j] = np.nan
            except Exception as e:
                print(f"    Error at E={E}, τ={tau}: {e}")
                results_matrix_xy[i, j] = np.nan
                results_matrix_yx[i, j] = np.nan

    return results_matrix_xy, results_matrix_yx, list(E_range), list(tau_range)


def plot_embedding_sensitivity_heatmap(results_xy, results_yx, E_range, tau_range, var1, var2, title, save_path=None):
    """Create heatmap showing CCM sensitivity to embedding parameters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot X→Y direction
    im1 = ax1.imshow(results_xy, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=1.0, origin='lower')
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
                color = 'white' if abs(results_xy[i, j]) > 0.5 else 'black'
                ax1.text(j, i, f'{results_xy[i, j]:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')

    plt.colorbar(im1, ax=ax1, label='CCM ρ')

    # Plot Y→X direction
    im2 = ax2.imshow(results_yx, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=1.0, origin='lower')
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
                color = 'white' if abs(results_yx[i, j]) > 0.5 else 'black'
                ax2.text(j, i, f'{results_yx[i, j]:.2f}', ha='center', va='center',
                         color=color, fontsize=8, fontweight='bold')

    plt.colorbar(im2, ax=ax2, label='CCM ρ')

    plt.suptitle(f'Embedding Parameter Sensitivity: {title}\n{var1} ↔ {var2}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved embedding sensitivity plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def calculate_false_nearest_neighbors(data, max_E=8, tau=1):
    """Calculate False Nearest Neighbors to determine optimal embedding dimension."""
    fnn_percentages = []

    for E in range(1, max_E + 1):
        try:
            n_points = len(data) - E * tau
            if n_points < 50:
                fnn_percentages.append(np.nan)
                continue

            # Create embedding
            embedding = np.zeros((n_points, E))
            for i in range(E):
                embedding[:, i] = data[i * tau:i * tau + n_points]

            # Calculate nearest neighbor distances (simplified FNN)
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=3).fit(embedding)  # Need 3 to get 2 nearest neighbors
            distances, indices = nbrs.kneighbors(embedding)

            # Calculate ratio of nearest neighbor distances
            # FNN metric: ratio of distance in (E+1) vs E dimensions
            if E < max_E:
                # Simplified: use ratio of 1st to 2nd nearest neighbor as proxy
                ratios = distances[:, 2] / (distances[:, 1] + 1e-10)  # Add small value to avoid division by zero
                threshold = 2.0  # Standard FNN threshold
                fnn_pct = np.mean(ratios > threshold) * 100
            else:
                fnn_pct = 0  # Assume no false neighbors at max dimension

            fnn_percentages.append(fnn_pct)

        except Exception as e:
            print(f"    FNN calculation error at E={E}: {e}")
            fnn_percentages.append(np.nan)

    return fnn_percentages


def calculate_mutual_information(data, max_tau=15):
    """Calculate mutual information to determine optimal time delay."""
    from sklearn.feature_selection import mutual_info_regression

    mutual_info = []

    for tau in range(1, max_tau + 1):
        try:
            if len(data) <= tau:
                mutual_info.append(np.nan)
                continue

            x = data[:-tau].reshape(-1, 1)
            y = data[tau:]

            # Calculate mutual information
            mi = mutual_info_regression(x, y, random_state=42)[0]
            mutual_info.append(mi)

        except Exception as e:
            print(f"    MI calculation error at τ={tau}: {e}")
            mutual_info.append(np.nan)

    return mutual_info


def plot_optimal_embedding_analysis(data_dict, var_name, title, save_path=None):
    """Plot FNN and MI analysis for optimal embedding parameter selection."""
    data = data_dict[var_name]

    # Calculate FNN and MI
    fnn_percentages = calculate_false_nearest_neighbors(data, max_E=8, tau=1)
    mutual_info = calculate_mutual_information(data, max_tau=15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot False Nearest Neighbors
    E_range = range(1, len(fnn_percentages) + 1)
    valid_fnn = [(e, fnn) for e, fnn in zip(E_range, fnn_percentages) if not np.isnan(fnn)]

    if valid_fnn:
        E_vals, fnn_vals = zip(*valid_fnn)
        ax1.plot(E_vals, fnn_vals, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Embedding Dimension (E)')
        ax1.set_ylabel('False Nearest Neighbors (%)')
        ax1.set_title(f'Optimal E Selection: {var_name}')
        ax1.grid(True, alpha=0.3)

        # Mark the optimal E (first significant drop or minimum)
        if len(fnn_vals) > 1:
            # Find first local minimum or significant drop
            optimal_E = E_vals[0]  # Default to E=1
            for i in range(1, len(fnn_vals)):
                if fnn_vals[i] < fnn_vals[i - 1] * 0.5:  # 50% drop
                    optimal_E = E_vals[i]
                    break

            ax1.axvline(optimal_E, color='red', linestyle='--', linewidth=2,
                        label=f'Optimal E={optimal_E}')
            ax1.legend()

    # Plot Mutual Information
    tau_range = range(1, len(mutual_info) + 1)
    valid_mi = [(tau, mi) for tau, mi in zip(tau_range, mutual_info) if not np.isnan(mi)]

    if valid_mi:
        tau_vals, mi_vals = zip(*valid_mi)
        ax2.plot(tau_vals, mi_vals, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Time Delay (τ)')
        ax2.set_ylabel('Mutual Information')
        ax2.set_title(f'Optimal τ Selection: {var_name}')
        ax2.grid(True, alpha=0.3)

        # Mark the first local minimum (optimal τ)
        if len(mi_vals) > 2:
            # Find first local minimum
            optimal_tau = tau_vals[0]  # Default to τ=1
            for i in range(1, len(mi_vals) - 1):
                if mi_vals[i] < mi_vals[i - 1] and mi_vals[i] < mi_vals[i + 1]:
                    optimal_tau = tau_vals[i]
                    break

            ax2.axvline(optimal_tau, color='red', linestyle='--', linewidth=2,
                        label=f'Optimal τ={optimal_tau}')
            ax2.legend()

    plt.suptitle(f'Embedding Parameter Optimization: {title}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved embedding optimization plot to: {save_path}")
    else:
        plt.show()
    plt.close()

    return fnn_percentages, mutual_info


def create_embedding_comparison_table(data_dict, var1, var2, E_tau_combinations, title):
    """Create comparison table showing dramatic differences with embedding parameters."""
    print(f"\n{'=' * 60}")
    print(f"Embedding Parameter Comparison: {title}")
    print(f"Testing {var1} ↔ {var2}")
    print(f"{'=' * 60}")
    print(f"{'E':>2} {'τ':>2} {'X→Y':>6} {'Y→X':>6} {'Interpretation'}")
    print(f"{'-' * 60}")

    results = []

    for E, tau in E_tau_combinations:
        try:
            result = run_ccm_direct(data_dict, var1, var2, E=E, tau=tau, lib_sizes=[250])

            if result is not None and len(result) > 0:
                final_row = result.iloc[-1]
                xy_corr = final_row.get('X:Y', np.nan)
                yx_corr = final_row.get('Y:X', np.nan)

                # Determine interpretation
                if np.isnan(xy_corr) or np.isnan(yx_corr):
                    interpretation = "Failed"
                elif abs(xy_corr) > 0.8 and abs(yx_corr) > 0.8:
                    interpretation = "Very strong (possibly over-connected)"
                elif abs(xy_corr) > 0.5 and abs(yx_corr) > 0.5:
                    interpretation = "Strong connection"
                elif abs(xy_corr) > 0.3 or abs(yx_corr) > 0.3:
                    interpretation = "Moderate connection"
                elif xy_corr < 0 and yx_corr < 0:
                    interpretation = "Independent (negative correlation)"
                else:
                    interpretation = "Weak/unclear"

                print(f"{E:>2} {tau:>2} {xy_corr:>6.2f} {yx_corr:>6.2f} {interpretation}")
                results.append((E, tau, xy_corr, yx_corr, interpretation))
            else:
                print(f"{E:>2} {tau:>2} {'N/A':>6} {'N/A':>6} Failed")
                results.append((E, tau, np.nan, np.nan, "Failed"))

        except Exception as e:
            print(f"{E:>2} {tau:>2} {'ERR':>6} {'ERR':>6} Error: {str(e)[:20]}")
            results.append((E, tau, np.nan, np.nan, f"Error: {str(e)[:20]}"))

    return results


def demonstrate_embedding_failures(data_dict, simulation_name, output_dir):
    """Demonstrate dramatic failures with wrong embedding parameters."""
    print(f"\n{'=' * 60}")
    print(f"EMBEDDING PARAMETER DEPENDENCY ANALYSIS: {simulation_name}")
    print(f"{'=' * 60}")

    # Test key relationships with different embedding parameters
    test_pairs = [('X1', 'X2'), ('X1', 'X4')]  # Strong connection and independence

    for var1, var2 in test_pairs:
        print(f"\nTesting {var1} ↔ {var2}...")

        # Define test combinations showing dramatic differences
        E_tau_combinations = [
            (1, 1),  # Under-embedding
            (2, 1),  # Still under-embedded
            (3, 1),  # Our standard (should be good)
            (4, 1),  # Slight over-embedding
            (6, 1),  # Over-embedding
            (3, 3),  # Different tau
            (3, 5),  # Large tau
        ]

        # Create comparison table
        results = create_embedding_comparison_table(data_dict, var1, var2,
                                                    E_tau_combinations, simulation_name)

        # Create sensitivity heatmap
        print(f"  Creating embedding sensitivity heatmap for {var1} ↔ {var2}...")
        results_xy, results_yx, E_range, tau_range = embedding_sensitivity_analysis(
            data_dict, var1, var2, E_range=range(1, 7), tau_range=range(1, 6))

        heatmap_path = os.path.join(output_dir,
                                    f"{simulation_name.lower().replace(' ', '_')}_embedding_sensitivity_{var1}_{var2}.png")
        plot_embedding_sensitivity_heatmap(results_xy, results_yx, E_range, tau_range,
                                           var1, var2, simulation_name, save_path=heatmap_path)

    # Optimal parameter analysis for each variable
    for var in ['X1', 'X2', 'X3', 'X4']:
        print(f"\n  Analyzing optimal parameters for {var}...")
        opt_path = os.path.join(output_dir, f"{simulation_name.lower().replace(' ', '_')}_optimal_params_{var}.png")
        fnn, mi = plot_optimal_embedding_analysis(data_dict, var, simulation_name, save_path=opt_path)


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

            # Get the maximum library size
            max_lib = result['LibSize'].max()

            # Get the row with the max library size
            row = result[result['LibSize'] == max_lib].iloc[0]

            # Determine which columns to use for causality values
            x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
            y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

            # If X:Y and Y:X aren't available, try to find other numeric columns
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


def save_results_to_csv(results, filename):
    """Save CCM results to CSV file."""
    data_rows = []

    for key, result in results.items():
        if result is not None:
            source, target = key.split('_')

            # Determine which columns to use for causality values
            x_to_y_col = 'X:Y' if 'X:Y' in result.columns else None
            y_to_x_col = 'Y:X' if 'Y:X' in result.columns else None

            # If X:Y and Y:X aren't available, try to find other numeric columns
            if x_to_y_col is None or y_to_x_col is None:
                numeric_cols = [col for col in result.columns if col != 'LibSize']
                if len(numeric_cols) >= 2:
                    x_to_y_col = numeric_cols[0]
                    y_to_x_col = numeric_cols[1]

            # Create a row for each library size
            for _, row in result.iterrows():
                data_dict = {
                    'Source': source,
                    'Target': target,
                    'Direction_X_Y': f"{source} → {target}",
                    'Direction_Y_X': f"{target} → {source}",
                    'Library_Size': row['LibSize']
                }

                # Add prediction values if available
                if x_to_y_col:
                    data_dict['X_predicts_Y'] = row[x_to_y_col]
                if y_to_x_col:
                    data_dict['Y_predicts_X'] = row[y_to_x_col]

                data_rows.append(data_dict)

    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Saved results to: {filename}")


def run_all_simulations():
    """Run all three simulations and compare results."""
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = f"causality_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "simulation_log.txt")

    # Capture ALL output to log file
    with LogCapture(log_file):
        print("Running Three Causality Simulations")
        print("=" * 60)
        print(f"Output will be saved to: {output_dir}")
        print(f"Log file: {log_file}")

        # Simulation 1: Linear Chain
        print("\n1. Running Simulation 1: Linear Chain (1 → 2 → 3 → 4)")
        print("-" * 50)
        data1 = generate_simulation_1(n_points=1500, strength=0.2, delay=3)

        ts_plot1 = os.path.join(output_dir, "sim1_timeseries.png")
        visualize_time_series(data1, "Simulation 1: Linear Chain (1 → 2 → 3 → 4)", save_path=ts_plot1)

        # Create phase space reconstruction plots
        phase_plot1 = os.path.join(output_dir, "sim1_phase_space.png")
        plot_phase_space_reconstruction(data1, "Simulation 1: Linear Chain", E=3, tau=1, save_path=phase_plot1)

        # Analyze phase space patterns
        analyze_phase_space_patterns(data1, "Simulation 1: Linear Chain", E=3, tau=1)

        # Demonstrate embedding dependency
        demonstrate_embedding_failures(data1, "Simulation 1", output_dir)

        print("  Analyzing causality...")
        results1 = analyze_all_pairs(data1, E=3, tau=1)
        expected1 = [(0, 1, "Strong"), (1, 2, "Strong"), (2, 3, "Strong")]  # X1→X2, X2→X3, X3→X4

        # Create causality matrix plot
        causality_plot1 = os.path.join(output_dir, "sim1_causality.png")
        create_causality_plot(results1, "Simulation 1: Linear Chain", expected1, save_path=causality_plot1)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results1, "Sim1", output_dir)

        # Print summary
        print_summary(results1, "Simulation 1", expected1)

        # Simulation 2: Hub with Outsider
        print("\n2. Running Simulation 2: Hub + Outsider (1 → 2 strong, 1 → 3 weak, 4 independent)")
        print("-" * 50)
        data2 = generate_simulation_2(n_points=1500, strong_strength=0.2, weak_strength=0.01, delay=3)

        ts_plot2 = os.path.join(output_dir, "sim2_timeseries.png")
        visualize_time_series(data2, "Simulation 2: Hub + Outsider", save_path=ts_plot2)

        # Create phase space reconstruction plots
        phase_plot2 = os.path.join(output_dir, "sim2_phase_space.png")
        plot_phase_space_reconstruction(data2, "Simulation 2: Hub + Outsider", E=3, tau=1, save_path=phase_plot2)

        # Analyze phase space patterns
        analyze_phase_space_patterns(data2, "Simulation 2: Hub + Outsider", E=3, tau=1)

        # Demonstrate embedding dependency
        demonstrate_embedding_failures(data2, "Simulation 2", output_dir)

        print("  Analyzing causality...")
        results2 = analyze_all_pairs(data2, E=3, tau=1)
        expected2 = [(0, 1, "Strong"), (0, 2, "Weak")]  # X1→X2 strong, X1→X3 weak

        causality_plot2 = os.path.join(output_dir, "sim2_causality.png")
        create_causality_plot(results2, "Simulation 2: Hub + Outsider", expected2, save_path=causality_plot2)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results2, "Sim2", output_dir)

        print_summary(results2, "Simulation 2", expected2)

        # Simulation 3: Complex Network
        print("\n3. Running Simulation 3: Complex Network (1 → 2 strong, 1 → 3 weak, 4 → 2 weak)")
        print("-" * 50)
        data3 = generate_simulation_3(n_points=1500, strong_strength=0.2, weak_strength=0.01, delay=3)

        ts_plot3 = os.path.join(output_dir, "sim3_timeseries.png")
        visualize_time_series(data3, "Simulation 3: Complex Network", save_path=ts_plot3)

        # Create phase space reconstruction plots
        phase_plot3 = os.path.join(output_dir, "sim3_phase_space.png")
        plot_phase_space_reconstruction(data3, "Simulation 3: Complex Network", E=3, tau=1, save_path=phase_plot3)

        # Analyze phase space patterns
        analyze_phase_space_patterns(data3, "Simulation 3: Complex Network", E=3, tau=1)

        # Demonstrate embedding dependency
        demonstrate_embedding_failures(data3, "Simulation 3", output_dir)

        print("  Analyzing causality...")
        results3 = analyze_all_pairs(data3, E=3, tau=1)
        expected3 = [(0, 1, "Strong"), (0, 2, "Weak"), (3, 1, "Weak")]  # X1→X2 strong, X1→X3 weak, X4→X2 weak

        causality_plot3 = os.path.join(output_dir, "sim3_causality.png")
        create_causality_plot(results3, "Simulation 3: Complex Network", expected3, save_path=causality_plot3)

        # Create convergence plots for all pairs
        create_all_convergence_plots(results3, "Sim3", output_dir)

        print_summary(results3, "Simulation 3", expected3)

        # Save results to CSV files
        save_results_to_csv(results1, os.path.join(output_dir, "sim1_results.csv"))
        save_results_to_csv(results2, os.path.join(output_dir, "sim2_results.csv"))
        save_results_to_csv(results3, os.path.join(output_dir, "sim3_results.csv"))

        # Final Comparison
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        print("Simulation 1 (Chain): Sequential causality with equal strength")
        print("Simulation 2 (Hub): One node influences multiple others with different strengths")
        print("Simulation 3 (Network): Multiple nodes contributing to same targets")
        print("\nKey insights:")
        print("- CCM should detect strongest direct relationships most clearly")
        print("- Indirect relationships (X1→X3 in Simulation 1) should be weaker")
        print("- Multiple influences on same target should be detectable but weaker individually")
        print("- Embedding parameters (E, τ) critically affect all CCM results")
        print("- Systematic parameter optimization essential for reliable causality detection")

        print(f"\nAll outputs saved to: {output_dir}")
        print(f"Log file: {log_file}")
        print(f"Embedding sensitivity analyses completed for all simulations.")

        # Summary of embedding dependency demonstration
        print(f"\n{'=' * 60}")
        print("EMBEDDING DEPENDENCY SUMMARY")
        print(f"{'=' * 60}")
        print("Generated analysis files:")
        print("• Embedding sensitivity heatmaps for key variable pairs")
        print("• Optimal parameter analysis (FNN and MI) for each variable")
        print("• Comparison tables showing dramatic parameter effects")
        print("• Visual demonstration of embedding dependency")
        print("This analysis proves CCM results are meaningless without proper embedding!")

        print(f"\nAll outputs saved to: {output_dir}")
        print(f"Log file: {log_file}")

    # After logging context is closed, print to console
    print(f"\nSimulation completed! All results saved to: {output_dir}")
    print(f"Check {log_file} for detailed log including any warnings.")
    print(f"Convergence plots have been created for all variable pairs.")


if __name__ == "__main__":
    run_all_simulations()