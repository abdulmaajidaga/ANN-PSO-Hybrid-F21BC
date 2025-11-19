import numpy as np
from main import main   # change to actual filename
import Utility.visualizer as v
import Utility.model_utils as model_utils
import os

NUM_RUNS = 5
BASE_DIR = "_Convergence_Test_3"

final_train_losses = []
test_losses = []

last_result = None  # to store final run for visualization

best_result = None
best_test_loss = float("inf")


for run in range(NUM_RUNS):
    print(f"\n========== RUN {run+1}/{NUM_RUNS} ==========\n")

    result = main()

    # Record metrics
    final_train_losses.append(result["final_train_loss"])
    test_losses.append(result["test_loss"])

    # Check if this is the best run (lowest test loss)
    if result["test_loss"] < best_test_loss:
        best_test_loss = result["test_loss"]
        best_result = result  # store everything needed


# Convert to numpy for statistics
final_train_losses = np.array(final_train_losses)
test_losses = np.array(test_losses)

final_train_loss_mean = final_train_losses.mean()
final_train_loss_std = final_train_losses.std()
final_test_loss_mean = test_losses.mean()
final_test_loss_std = test_losses.std()


print("\n==================== SUMMARY ====================")
print(f"Final Train Loss Mean: {final_train_losses.mean():.6f}, "
      f"Std: {final_train_losses.std():.6f}")
print(f"Test Loss Mean: {test_losses.mean():.6f}, "
      f"Std: {test_losses.std():.6f}")
print("=================================================")

# ----------------------------------------------------
# VISUALIZE USING LAST RUN
# ----------------------------------------------------
print("\n=== Creating visualizations using LAST RUN ===\n")

optimizer = best_result["optimizer"]
ann_pso_bridge = best_result["ann_pso_bridge"]
model_template = best_result["model_template"]
pso_params = best_result["params"]
y_test = best_result["y_test"]
y_test_predictions = best_result["y_test_predictions"]


visualizer = v.Visualizer(
    pso=optimizer,
    layers=[8,16,16,1],           # same as your original
    pso_params=pso_params,                # same as your settings
    num_particles=30,
    num_iterations=200,
    num_informants=6,
    loss_function="mae",
    train_loss=best_result["final_train_loss"],
    test_loss=best_result["test_loss"],
    base_dir = BASE_DIR
)

test_folder = visualizer.record_test()

model_utils.plot_predictions(
    y_test,
    y_test_predictions,
    test_folder=test_folder
)

"""Create a txt or md file summarizing the parameters."""
file_path = os.path.join(test_folder, "_test_summary.txt")
with open(file_path, "w") as f:
    f.write(f"Final Train Loss Mean: {final_train_loss_mean:.6f}\n")
    f.write(f"Final Train Loss std: {final_train_loss_std:.6f}\n")
    f.write(f"Test Loss Mean: {final_test_loss_mean:.6f}\n")
    f.write(f"Test Loss std: {final_test_loss_std:.6f}\n")
    
print(f"[INFO] Parameters saved to: {file_path}")

