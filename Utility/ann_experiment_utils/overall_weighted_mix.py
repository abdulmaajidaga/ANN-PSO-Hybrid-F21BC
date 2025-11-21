import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# USER CONSTANTS (ADJUST WEIGHTS HERE)
# ==========================================
WEIGHT_LOSS = 0.8            # Weight for Performance (mae_mean)
WEIGHT_GENERALIZABILITY = 0.2 # Weight for Robustness (gap_mean)

# Files
FILE_MAIN_DATA = 'Grid_Search_Results/2000_10_grid_results.csv'  # The 2000_10 file (Primary)
FILE_GEN_DATA = 'Grid_Search_Results_Gen/grid_results_summary.csv'   # The summary file (Lookup for gap)

# Columns
LOSS_COL = 'mae_mean'   # From 2000_10 file
GEN_COL = 'gap_mean'    # From summary file

def plot_weighted_heatmap(df, metric_col):
    """
    Generates a heatmap styled exactly like vizplot.py.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Create Pivot Table
    pivot_table = df.pivot(index="width", columns="depth", values=metric_col)
    
    # Plot Heatmap
    # cmap="RdYlGn": Green is High (Good Score), Red is Low (Bad Score)
    ax = sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="RdYlGn", 
                     linewidths=.5, cbar_kws={'label': 'Weighted Score (Higher is Better)'})
    
    ax.set_title(f"Weighted Score Heatmap\n(Loss: {WEIGHT_LOSS} | Gen: {WEIGHT_GENERALIZABILITY})", 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Depth (Hidden Layers)", fontsize=12)
    ax.set_ylabel("Width", fontsize=12)
    
    ax.invert_yaxis()

    outfile = 'weighted_score_heatmap.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {outfile}")

def find_best_weighted_config(file_main, file_gen, w_loss, w_gen):
    try:
        # 1. Load Files
        print(f"Loading main data: {file_main}")
        df_main = pd.read_csv(file_main)
        
        # --- EXPLICIT FILTER: Keep only Depth 1 and 2 ---
        print("Filtering data to keep only Depth 1 and 2...")
        df_main = df_main[df_main['depth'].isin([1, 2])]
        
        if df_main.empty:
            raise ValueError("Error: Filtered data is empty. Check if the file contains Depth 1 or 2.")

        print(f"Loading generalizability data: {file_gen}")
        df_gen = pd.read_csv(file_gen)
        
        # 2. Selective Merge
        print("Merging data...")
        df_gen_subset = df_gen[['depth', 'width', GEN_COL]]
        merged_df = pd.merge(df_main, df_gen_subset, on=['depth', 'width'], how='left')
        
        # Check for missing gap values
        if merged_df[GEN_COL].isnull().any():
            print("Warning: Some configurations in 2000_10 were not found in the summary file.")
            merged_df = merged_df.dropna(subset=[GEN_COL])

        print(f"Total configurations to analyze: {len(merged_df)}")
        print("-" * 40)

        # 3. Normalize metrics & Invert Logic
        # Scaler transforms data to range [0, 1].
        # Since we want LESSER loss to be MORE score, we do (1 - scaled_value).
        scaler = MinMaxScaler()
        
        # Normalize Loss (Lower MAE -> Higher Score)
        raw_loss_scaled = scaler.fit_transform(merged_df[[LOSS_COL]])
        merged_df['loss_score'] = 1 - raw_loss_scaled
        
        # Normalize Gen Gap (Lower Gap -> Higher Score)
        raw_gen_scaled = scaler.fit_transform(merged_df[[GEN_COL]])
        merged_df['gen_score'] = 1 - raw_gen_scaled

        # 4. Calculate Weighted Score (Higher is Better)
        merged_df['weighted_score'] = (merged_df['loss_score'] * w_loss) + (merged_df['gen_score'] * w_gen)

        # 5. Sort and Find Best (Sort Descending because Higher is Better)
        df_sorted = merged_df.sort_values(by='weighted_score', ascending=False).reset_index(drop=True)
        best_config = df_sorted.iloc[0]

        # 6. Display Results
        print(f"WEIGHTS: Loss ({w_loss}) | Generalizability ({w_gen})")
        print("BEST OVERALL CONFIGURATION (Depth 1 & 2 Only):")
        print(f" > Depth: {int(best_config['depth'])}")
        print(f" > Width: {int(best_config['width'])}")
        print(f" > {LOSS_COL} (Main): {best_config[LOSS_COL]:.4f}")
        print(f" > {GEN_COL} (Lookup): {best_config[GEN_COL]:.4f}")
        print(f" > Weighted Score:  {best_config['weighted_score']:.4f} (Max 1.0)")
        
        # Save CSV
        output_filename = 'combined_weighted_results.csv'
        df_sorted.to_csv(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")

        # 7. Visualize
        plot_weighted_heatmap(merged_df, 'weighted_score')

    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
    except KeyError as e:
        print(f"Error: Column missing. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    find_best_weighted_config(FILE_MAIN_DATA, FILE_GEN_DATA, WEIGHT_LOSS, WEIGHT_GENERALIZABILITY)