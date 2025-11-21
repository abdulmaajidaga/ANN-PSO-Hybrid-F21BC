import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

# Import your actual main module
import main as main_module 

# ============================================================================
# CONFIGURATION
# ============================================================================
DEPTHS = range(1, 5) 
WIDTHS = [4, 8, 16, 32, 64, 128]
NUM_RUNS_PER_ARCH = 10

# --- SMART THREADING CONFIG ---
FAST_WORKERS = 12      # For Width < 64 (Light)
SAFE_WORKERS = 1       # For Width >= 64 (Heavy - Prevents RAM Crash)
HEAVY_WIDTH_THRESHOLD = 64

# --- METRIC TOGGLE ---
# True: Measure Gap (|Test - Train|)
# False: Measure raw Test MAE
ENABLE_GENERALIZABILITY_METRICS = True 

OUTPUT_DIR = "Grid_Search_Results_Gen" if ENABLE_GENERALIZABILITY_METRICS else "Grid_Search_Results_MAE"
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
            
        # --- METRIC EXTRACTION LOGIC ---
        train_loss = result.get("final_train_loss", np.nan)
        test_loss = result.get("test_loss", np.nan)
        activations = result.get("activations", "Unknown")
        
        # Calculate Gap if needed
        gap = abs(test_loss - train_loss) if (not np.isnan(train_loss) and not np.isnan(test_loss)) else np.nan
        
        # Force cleanup
        gc.collect()

        return {
            "depth": depth,
            "width": width,
            "mae": test_loss,       # Always store raw MAE
            "train_loss": train_loss, # Store train loss
            "gap": gap,             # Store gap
            "activations": activations,
            "layers": str(current_layers),
            "status": "success"
        }
    except Exception as e:
        return {
            "depth": depth,
            "width": width,
            "mae": np.nan,
            "gap": np.nan,
            "error": str(e),
            "status": "failed"
        }

def run_batch(tasks_config, num_workers, batch_name):
    """Helper to run a specific list of tasks with a set number of workers"""
    results = []
    futures = []
    
    if not tasks_config:
        return results

    print(f"\n--- Starting {batch_name} Batch ({len(tasks_config)} runs) ---")
    print(f"    Using {num_workers} Workers for stability.")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for (d, w, r) in tasks_config:
            futures.append(executor.submit(worker_task, d, w, r))
            
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res["status"] == "success":
                results.append(res)
            else:
                print(f"[ERROR] Run Failed: {res['error']}")
            
            # Simple progress bar
            if (i + 1) % 5 == 0:
                val = res['gap'] if ENABLE_GENERALIZABILITY_METRICS else res['mae']
                label = "Gap" if ENABLE_GENERALIZABILITY_METRICS else "MAE"
                print(f"    [{i+1}/{len(tasks_config)}] completed. Last {label}: {val:.4f}")

    return results

# ============================================================================
# EXECUTION LOOP
# ============================================================================
def run_grid_search():
    start_time = time.time()
    
    # 1. Sort tasks into Light (Fast) and Heavy (Safe)
    light_tasks = []
    heavy_tasks = []
    
    total_runs = len(DEPTHS) * len(WIDTHS) * NUM_RUNS_PER_ARCH
    mode_str = "GENERALIZABILITY (Gap)" if ENABLE_GENERALIZABILITY_METRICS else "PERFORMANCE (MAE)"
    
    print("="*80)
    print(f"STARTING SMART-GRID SEARCH - MODE: {mode_str}")
    print(f"Total Runs: {total_runs}")
    print(f"Heavy Threshold: Width >= {HEAVY_WIDTH_THRESHOLD}")
    print("="*80)

    for depth in DEPTHS:
        for width in WIDTHS:
            for run in range(NUM_RUNS_PER_ARCH):
                task = (depth, width, run)
                if width >= HEAVY_WIDTH_THRESHOLD:
                    heavy_tasks.append(task)
                else:
                    light_tasks.append(task)
    
    # 2. Run Light Tasks (High Parallelism)
    results_data = []
    if light_tasks:
        results_data.extend(run_batch(light_tasks, FAST_WORKERS, "LIGHT"))
        
    # 3. Run Heavy Tasks (Low Parallelism to save RAM)
    if heavy_tasks:
        results_data.extend(run_batch(heavy_tasks, SAFE_WORKERS, "HEAVY"))

    elapsed = time.time() - start_time
    print(f"\nSearch Finished in {elapsed/60:.2f} minutes.")

    # ============================================================================
    # AGGREGATION
    # ============================================================================
    if not results_data:
        print("No results collected.")
        return pd.DataFrame()

    raw_df = pd.DataFrame(results_data)
    
    # Save RAW data (Important for activations)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "grid_results_raw.csv"), index=False)

    # Group by Depth/Width to get Mean/Std
    agg_df = raw_df.groupby(['depth', 'width']).agg(
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        gap_mean=('gap', 'mean'),
        gap_std=('gap', 'std')
    ).reset_index()

    # Save Summary Results
    agg_df.to_csv(os.path.join(OUTPUT_DIR, "grid_results_summary.csv"), index=False)
    print(f"Results saved to {OUTPUT_DIR}")

    return agg_df

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(df):
    if df.empty: return

    sns.set_style("whitegrid")
    
    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # --- TOGGLE METRIC SELECTION ---
    if ENABLE_GENERALIZABILITY_METRICS:
        plot_metric = "gap_mean"
        plot_title = "Generalizability Heatmap (Mean Gap)"
        cbar_label = "Gap (|Test - Train|)"
        ylabel_line = "Mean Generalization Gap"
    else:
        plot_metric = "mae_mean"
        plot_title = "Topology Heatmap (Mean Test MAE)"
        cbar_label = "Test MAE"
        ylabel_line = "Mean Test MAE"

    # --- PLOT 1: HEATMAP ---
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_table = df.pivot(index="width", columns="depth", values=plot_metric)
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax1, cbar_kws={'label': cbar_label})
    ax1.set_title(plot_title, fontsize=16, fontweight='bold')
    ax1.set_xlabel("Depth (Hidden Layers)", fontsize=12)
    ax1.set_ylabel("Width", fontsize=12)
    ax1.invert_yaxis()

    # --- PLOT 2: INTERACTION LINES ---
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=df, x="depth", y=plot_metric, hue="width", palette="magma", marker="o", linewidth=2, ax=ax2)
    ax2.set_title(f"Scaling Analysis: {ylabel_line}", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Depth", fontsize=12)
    ax2.set_ylabel(ylabel_line, fontsize=12)
    
    ax2.legend(title="Width", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "topology_analysis.png")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graphs saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Windows multiprocessing requires this protection
    agg_df = run_grid_search()
    plot_results(agg_df)