import os
import numpy as np
import matplotlib.pyplot as plt


class MultiExperimentVisualizer:
    def __init__(self, experiments_dict, base_dir="_Test_Results"):
        """
        Parameters
        ----------
        experiments_dict : dict
            {
                "Experiment Name 1": optimizer_1,
                "Experiment Name 2": optimizer_2,
                ...
            }
            Each optimizer must contain:
            - gbest_value_history
            - mean_fitness_history
            - particle_history (list of particle arrays)
        """
        self.experiments = experiments_dict
        self.base_dir = base_dir

        self.plots_dir = os.path.join(self.base_dir, "Plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    # =====================================================
    # 1) GLOBAL BEST FITNESS COMPARISON
    # =====================================================
    def plot_gbest_convergence(self):
        plt.figure(figsize=(9, 6))

        for name, opt in self.experiments.items():
            plt.plot(opt.gbest_value_history, label=name)

        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("Global Best Fitness Convergence Across Experiments")
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(self.plots_dir, "gbest_comparison.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] Global Best Comparison -> {save_path}")

    # =====================================================
    # 2) MEAN FITNESS COMPARISON (NO STD)
    # =====================================================
    def plot_population_fitness_convergence(self):
        plt.figure(figsize=(9, 6))

        for name, opt in self.experiments.items():
            mean = np.array(opt.mean_fitness_history)
            plt.plot(mean, label=name)

        plt.xlabel("Iteration")
        plt.ylabel("Mean Fitness")
        plt.title("Population Mean Fitness Convergence Across Experiments")
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(self.plots_dir, "mean_fitness_comparison.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] Mean Fitness Comparison -> {save_path}")

    # =====================================================
    # 3) MEAN POSITION CONVERGENCE
    # =====================================================
    def plot_position_convergence(self):
        plt.figure(figsize=(9, 6))

        for name, opt in self.experiments.items():
            # Compute mean scalar position per iteration
            mean_positions = [np.mean(p, axis=0) for p in opt.particle_history]
            mean_positions = np.array(mean_positions)
            mean_scalar = np.mean(mean_positions, axis=1)

            plt.plot(mean_scalar, label=name)

        plt.xlabel("Iteration")
        plt.ylabel("Mean Particle Position")
        plt.title("Mean Position Convergence Across Experiments")
        plt.grid(True)
        plt.legend()

        save_path = os.path.join(self.plots_dir, "position_convergence_comparison.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] Mean Position Comparison -> {save_path}")

    # =====================================================
    # RUN ALL PLOTS
    # =====================================================
    def generate_all(self):
        self.plot_gbest_convergence()
        self.plot_population_fitness_convergence()
        self.plot_position_convergence()
        print("\n[INFO] All multi-experiment comparison plots generated.\n")
