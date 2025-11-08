import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

class Visualizer:
    def __init__(self, pso, layers=None, pso_params=None, num_particles=None,
                 num_iterations=None, num_informants=None, loss_function=None,
                 intial_scaled_loss=None, final_scaled_loss=None, final_real_loss=None, test_real_loss = None,
                 base_dir="Test_Results"):
        """
        Initialize visualizer with PSO object and experiment metadata.
        """
        self.pso = pso
        self.layers = layers
        self.pso_params = pso_params
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.num_informants = num_informants
        self.loss_function = loss_function
        self.intial_scaled_loss = intial_scaled_loss
        self.final_scaled_loss = final_scaled_loss
        self.final_real_loss = final_real_loss
        self.test_real_loss = test_real_loss
        self.base_dir = base_dir


    def _write_params_file(self):
        """Create a txt or md file summarizing the parameters."""
        file_path = os.path.join(self.test_dir, "_parameters.txt")
        with open(file_path, "w") as f:
            f.write("# PSO–ANN Test Configuration\n\n")
            f.write(f"**Layers:** {self.layers}\n")
            f.write(f"**Num Particles:** {self.num_particles}\n")
            f.write(f"**Num Iterations:** {self.num_iterations}\n")
            f.write(f"**Num Informants:** {self.num_informants}\n")
            f.write(f"**Loss Function:** {self.loss_function}\n\n")
            f.write("## PSO Parameters\n")
            if self.pso_params:
                for k, v in self.pso_params.items():
                    f.write(f"- {k}: {v}\n\n")
                    
            f.write(f"Initial Best Loss (un-scaled): {self.intial_scaled_loss:.6f}\n")
            f.write(f"Final Best Loss (un-scaled): {self.final_scaled_loss:.6f}\n")
            f.write(f"Final Best Loss (scaled): {self.final_real_loss:.6f}\n")
            f.write(f"Test Set Loss (scaled): {self.test_real_loss:.6f}\n")
            
        print(f"[INFO] Parameters saved to: {file_path}")

    # -------------------------------------------------------------
    # PLOTS — Saved instead of shown
    # -------------------------------------------------------------     
        
    def animate_pso_pca_gif(self, filename="_pso_animation.gif", fps=25):
        """Generate and save a GIF animation of particle swarm movement (PCA projection)."""
        print("[INFO] Generating PSO PCA animation...")

        # --- Prepare PCA projection ---
        all_particles = np.vstack(self.pso.particle_history)
        pca = PCA(n_components=2)
        pca.fit(all_particles)

        proj_all = pca.transform(all_particles)
        x_min, x_max = proj_all[:, 0].min(), proj_all[:, 0].max()
        y_min, y_max = proj_all[:, 1].min(), proj_all[:, 1].max()

        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], c="blue", alpha=0.6)
        gb_point, = ax.plot([], [], "r*", markersize=10, label="Global Best")

        def init():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title("PSO Swarm Dynamics (PCA Projection)")
            ax.legend()
            return scat, gb_point

        def update(frame):
            particles = pca.transform(self.pso.particle_history[frame])
            gbest = pca.transform(self.pso.Gbest_position_history[frame].reshape(1, -1))
            scat.set_offsets(particles)
            gb_point.set_data(gbest[0, 0], gbest[0, 1])
            ax.set_title(f"Iteration {frame + 1}")
            return scat, gb_point

        ani = FuncAnimation(
            fig, update, frames=min(len(self.pso.particle_history), 100),
            init_func=init, blit=False, repeat=False
        )

        # --- Save as GIF ---
        gif_path = os.path.join(self.test_dir, filename)
        try:
            ani.save(gif_path, writer="pillow", fps=fps)
            print(f"[SAVED] PSO animation GIF -> {gif_path}")
        except Exception as e:
            print(f"[ERROR] Could not save animation GIF: {e}")
            print("You may need to install Pillow or ImageMagick.")

        plt.close(fig)


    def plot_fitness_convergence(self):
        """Plot and save global best and mean fitness convergence."""
        plt.figure(figsize=(8, 5))
        #plt.plot(self.pso.Gbest_value_history, label="Global Best")
        plt.plot(self.pso.mean_fitness_history, label="Mean Fitness", linestyle="--")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (MSE)")
        plt.title("PSO Convergence Curve")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.test_dir, "convergence_curve.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Convergence plot -> {save_path}")


    def plot_mean_position_convergence(self, component_wise=False):
        """
        Plot the convergence of the swarm in terms of average particle position.
        
        Parameters
        ----------
        component_wise : bool, optional (default=False)
            If True, plots each dimension of the mean position vector separately.
            If False, plots the L2 norm (magnitude) of the mean position over time.
        """
        mean_positions = [np.mean(particles, axis=0) for particles in self.pso.particle_history]
        mean_positions = np.array(mean_positions)  # shape: (iterations, dimensions)

        plt.figure(figsize=(8, 5))

        if component_wise and mean_positions.shape[1] <= 10:
            # Plot each dimension separately (only if reasonably few dimensions)
            for d in range(mean_positions.shape[1]):
                plt.plot(mean_positions[:, d], label=f"Dim {d+1}")
            plt.ylabel("Mean Position (per Dimension)")
            plt.title("Component-wise Mean Particle Position Convergence")
            plt.legend()
        else:
            # Plot the L2 norm (magnitude)
            norms = np.linalg.norm(mean_positions, axis=1)
            plt.plot(norms, color="teal")
            plt.ylabel("‖Mean Position‖ (L2 Norm)")
            plt.title("Mean Particle Position Convergence (Magnitude)")

        plt.xlabel("Iteration")
        plt.grid(True)

        save_path = os.path.join(self.test_dir, "mean_position_convergence.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Mean position convergence plot -> {save_path}")


    def plot_swarm_diversity(self):
        """Plot and save average swarm diversity across iterations."""
        diversities = []
        for particles in self.pso.particle_history:
            mean_pos = np.mean(particles, axis=0)
            diversity = np.mean(np.linalg.norm(particles - mean_pos, axis=1))
            diversities.append(diversity)

        plt.figure(figsize=(8, 5))
        plt.plot(diversities, color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Average Diversity")
        plt.title("Swarm Diversity Over Time")
        plt.grid(True)

        save_path = os.path.join(self.test_dir, "swarm_diversity.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Swarm diversity plot -> {save_path}")

    def plot_velocity_magnitude(self):
        """Plot and save average particle velocity magnitude."""
        avg_vel = [np.mean(np.linalg.norm(v, axis=1)) for v in self.pso.velocity_history]
        plt.figure(figsize=(8, 5))
        plt.plot(avg_vel, color="purple")
        plt.xlabel("Iteration")
        plt.ylabel("Average Velocity Magnitude")
        plt.title("Swarm Velocity Magnitude Over Time")
        plt.grid(True)

        save_path = os.path.join(self.test_dir, "velocity_magnitude.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Velocity magnitude plot -> {save_path}")


    def record_test(self, base_dir = "_Test_Results" ):
        
        # --- Create unique test folder ---
        os.makedirs(base_dir, exist_ok=True)
        existing_tests = [d for d in os.listdir(base_dir) if d.startswith("Test_")]
        next_index = len(existing_tests) + 1
        self.test_dir = os.path.join(base_dir, f"Test_{next_index}")
        os.makedirs(self.test_dir, exist_ok=True)

        # --- Write parameters file ---
        self._write_params_file()
        self.plot_fitness_convergence()
        self.plot_mean_position_convergence()
        self.plot_swarm_diversity()
        self.plot_velocity_magnitude()
        self.animate_pso_pca_gif()
        
        return self.test_dir