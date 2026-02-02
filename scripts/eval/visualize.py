import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"


def find_csv_file(result_subdir):
    # Look for csv files
    csv_files = list(result_subdir.glob("*.csv"))
    
    if not csv_files:
        return None
    
    for csv_file in csv_files:
        if 'all_horizons' in csv_file.name.lower() or 'baseline' in csv_file.name.lower():
            return csv_file
    
    return csv_files[0]


def load_results(result_subdir, model_name):
    csv_file = find_csv_file(result_subdir)
    
    if csv_file is None:
        print(f"No CSV file found in {result_subdir}, skipping.")
        return None
    
    print(f"Loading {model_name} from {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'horizon' not in df.columns:
            print(f"Warning: 'horizon' column not found in {csv_file}, skipping.")
            return None
        
        if 'test_rmse' not in df.columns:
            print(f"Warning: 'test_rmse' column not found in {csv_file}, skipping.")
            return None
        
        # Extract horizon and test_rmse
        results = df[['horizon', 'test_rmse']].copy()
        results['model'] = model_name
        
        return results
    
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None


def plot_rmse_comparison():
    
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    
    all_results = []
    
    for subdir in RESULTS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        #Skip the plots directory itself
        if subdir.name == 'plots':
            continue
        
        model_name = subdir.name
        results = load_results(subdir, model_name)
        
        if results is not None:
            all_results.append(results)
    
    if not all_results:
        print("No valid results found to plot.")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    #Plot each model
    for model_name in combined_df['model'].unique():
        model_data = combined_df[combined_df['model'] == model_name].sort_values('horizon')
        plt.plot(
            model_data['horizon'],
            model_data['test_rmse'],
            marker='o',
            linewidth=2,
            label=model_name,
            markersize=6
        )
    
    plt.xlabel('Forecast Horizon (hours)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('Model Performance Comparison Across Forecast Horizons', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    #Save the plot
    output_path = OUTPUT_DIR / "rmse_vs_horizon_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    #Also save as PDF
    output_path_pdf = OUTPUT_DIR / "rmse_vs_horizon_comparison.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_path_pdf}")
    
    plt.show()
    
    #Print summary statistics
    print("\n Summary Statistics:")
    summary = combined_df.groupby('model')['test_rmse'].agg(['mean', 'min', 'max'])
    print(summary)


def main():
    plot_rmse_comparison()


if __name__ == "__main__":
    main()