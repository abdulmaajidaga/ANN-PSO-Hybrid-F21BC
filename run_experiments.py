import numpy as np
from main import main
import Utility.visualizer as v
import Utility.model_utils as model_utils
from Utility.multi_experiment_visualizer import MultiExperimentVisualizer
import os


# ============================================================
# Load experiment params from TXT
# ============================================================

def load_experiments_from_txt(file_path):
    experiments = []
    current_name = None
    current_params = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # skip empty lines or comments
            if not line or line.startswith("#"):
                continue

            # New experiment block
            if line.startswith("[") and line.endswith("]"):
                # Save previous experiment (if any)
                if current_name is not None:
                    experiments.append((current_name, current_params))
                
                current_name = line[1:-1]   # remove [ ]
                current_params = {}
                continue

            # Parameter line: key = value
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = float(value.strip())
                current_params[key] = value

        # Append last block if exists
        if current_name is not None:
            experiments.append((current_name, current_params))

    return experiments


# =================== CONFIG =======================
EXPERIMENT_FILE = "Experiments/experiment_4.txt"
NUM_RUNS = 5
BASE_DIR = "EXPERIMENT_4"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Load experiments dynamically
EXPERIMENTS = load_experiments_from_txt(EXPERIMENT_FILE)

best_results = {}

for exp_name, exp_params in EXPERIMENTS:

    print(f"\n==================== RUNNING {exp_name} ====================\n")

    exp_folder = os.path.join(BASE_DIR, exp_name)
    ensure_dir(exp_folder)

    final_train_losses = []
    test_losses = []

    best_result = None
    best_test_loss = float("inf")

    for run in range(NUM_RUNS):
        print(f"\n========== {exp_name} | RUN {run + 1}/{NUM_RUNS} ==========\n")

        result = main(exp_params)

        final_train_losses.append(result["final_train_loss"])
        test_losses.append(result["test_loss"])

        if result["test_loss"] < best_test_loss:
            best_test_loss = result["test_loss"]
            best_result = result

    final_train_losses = np.array(final_train_losses)
    test_losses = np.array(test_losses)

    summary_path = os.path.join(exp_folder, "_experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"===== {exp_name} =====\n\n")
        f.write(f"Final Train Loss Mean: {final_train_losses.mean():.6f}\n")
        f.write(f"Final Train Loss Std:  {final_train_losses.std():.6f}\n")
        f.write(f"Test Loss Mean:        {test_losses.mean():.6f}\n")
        f.write(f"Test Loss Std:         {test_losses.std():.6f}\n")
        f.write("\nBest Run Test Loss:\n")
        f.write(f"{best_test_loss:.6f}\n")

    print(f"\n[INFO] Summary saved to {summary_path}")

    print(f"\n=== Creating visualizations for BEST RUN of {exp_name} ===\n")

    optimizer = best_result["optimizer"]
    y_test = best_result["y_test"]
    y_test_predictions = best_result["y_test_predictions"]

    base_folder_override = os.path.join(exp_folder, "best_run")

    visualizer = v.Visualizer(
        pso=optimizer,
        layers=best_result["layers"],
        pso_params=exp_params,
        num_particles=30,
        num_iterations=200,
        num_informants=6,
        loss_function="mae",
        train_loss=best_result["final_train_loss"],
        test_loss=best_result["test_loss"],
        base_dir=base_folder_override,
    )

    test_folder = visualizer.record_test()

    model_utils.plot_predictions(
        y_test,
        y_test_predictions,
        test_folder=test_folder
    )

    print(f"[INFO] Visualizations saved to {test_folder}")

    best_results[exp_name] = optimizer

print("\n==================== ALL EXPERIMENTS COMPLETED ====================\n")

viz = MultiExperimentVisualizer(best_results, base_dir=BASE_DIR)
viz.generate_all()
