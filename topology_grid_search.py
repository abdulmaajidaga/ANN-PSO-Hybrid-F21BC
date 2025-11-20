import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

# Import your actual main module
import main as main_module 

# ============================================================================
# CONFIGURATION
# ============================================================================
DEPTHS = range(1, 9) 
WIDTHS = [4, 8, 16, 32, 64, 128]
NUM_RUNS_PER_ARCH = 10  
NUM_WORKERS = 8

OUTPUT_DIR = "Grid_Search_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# UTILITIES
# ============================================================================
@contextmanager
def suppress_stdout():
    """Prevents parallel processes from spamming the console"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def worker_task(depth, width, run_id):
    """
    This function runs inside a separate process.
    """
    try:
        # Construct layer list: Input(8) + [Hidden]*Depth + Output(1)
        current_layers = [8] + [width] * depth + [1]
        
        # Inject into main module specific to this process
        main_module.LAYERS = current_layers
        
        # Run main() silently
        with suppress_stdout():
            result = main_module.main()
            
        return {
            "depth": depth,
            "width": width,
            "mae": result["test_loss"],
            "layers": str(current_layers),
            "status": "success"
        }
    except Exception as e:
        return {
            "depth": depth,
            "width": width,
            "mae": np.nan,
            "error": str(e),
            "status": "failed"
        }

# ============================================================================
# EXECUTION LOOP
# ============================================================================
def run_grid_search():
    results_data = []
    tasks = []
    
    total_combinations = len(DEPTHS) * len(WIDTHS) * NUM_RUNS_PER_ARCH
    
    print("="*80)
    print(f"STARTING PARALLEL GRID SEARCH")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Total Runs: {total_combinations}")
    print("="*80)
    
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 1. Submit all tasks
        for depth in DEPTHS:
            for width in WIDTHS:
                for run in range(NUM_RUNS_PER_ARCH):
                    future = executor.submit(worker_task, depth, width, run)
                    tasks.append(future)
        
        # 2. Monitor progress
        completed = 0
        for future in as_completed(tasks):
            res = future.result()
            completed += 1
            
            if res["status"] == "success":
                # We flatten the result immediately for storage
                results_data.append(res)
                # Optional: Print progress every 10 runs
                if completed % 10 == 0:
                    print(f"[{completed}/{total_combinations}] Completed. Last MAE: {res['mae']:.4f}")
            else:
                print(f"[ERROR] Task Failed: {res['error']}")

    elapsed = time.time() - start_time
    print(f"\nSearch Finished in {elapsed/60:.2f} minutes.")

    # ============================================================================
    # AGGREGATION
    # ============================================================================
    raw_df = pd.DataFrame(results_data)
    
    # Group by Depth/Width to get Mean/Std
    agg_df = raw_df.groupby(['depth', 'width']).agg(
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        mae_min=('mae', 'min')
    ).reset_index()

    # Save Results
    agg_df.to_csv(os.path.join(OUTPUT_DIR, "grid_results.csv"), index=False)
    print(f"Results saved to {OUTPUT_DIR}")

    return agg_df

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(df):
    if df.empty: return

    sns.set_style("whitegrid")
    
    # FIX 1: Increased figure width (from 20 to 22) to give the legend room
    fig = plt.figure(figsize=(22, 9))
    
    # FIX 2: Add wspace (width space) to separate the two plots
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # --- PLOT 1: HEATMAP ---
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_table = df.pivot(index="width", columns="depth", values="mae_mean")
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax1, cbar_kws={'label': 'Test MAE'})
    ax1.set_title("Topology Heatmap (Mean MAE)", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Depth (Hidden Layers)", fontsize=12)
    ax1.set_ylabel("Width", fontsize=12)
    ax1.invert_yaxis()

    # --- PLOT 2: INTERACTION LINES ---
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=df, x="depth", y="mae_mean", hue="width", palette="magma", marker="o", linewidth=2, ax=ax2)
    ax2.set_title("Performance Scaling: Depth vs Width", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Depth", fontsize=12)
    ax2.set_ylabel("Mean Test MAE", fontsize=12)
    
    # FIX 3: Explicitly place legend outside and ensure it has an anchor
    ax2.legend(title="Width", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # FIX 4: Smart layout adjustment
    # This automatically fixes overlaps and makes room for labels
    plt.tight_layout()

    # Save logic
    plot_path = os.path.join(OUTPUT_DIR, "topology_analysis.png")
    
    # FIX 5: bbox_inches='tight' ensures the external legend is included in the saved file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graphs saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Windows multiprocessing requires this protection
    agg_df = run_grid_search()
    plot_results(agg_df)