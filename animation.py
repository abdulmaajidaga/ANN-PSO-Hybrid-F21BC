# save_climbing_animation.py
import numpy as np
import pso  # Your modified pso.py file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sys # To show progress

# --- 1. THE MATH PROBLEM ---

BOUND_LOW = -5.12
BOUND_HIGH = 5.12

def rastrigin_valley(particles):
    """
    The original Rastrigin function (a valley).
    The PSO will MINIMIZE this.
    """
    x = particles[:, 0]
    y = particles[:, 1]
    A = 10
    return (A * 2) + \
           (x**2 - A * np.cos(2 * np.pi * x)) + \
           (y**2 - A * np.cos(2 * np.pi * y))

def rastrigin_hill(particles):
    """
    The inverted Rastrigin function (a hill).
    We will PLOT this.
    """
    return -rastrigin_valley(particles)

# --- MODIFIED: Added Ackley Functions ---
def ackley_valley(particles):
    """
    The Ackley function (a valley with ripples).
    The PSO will MINIMIZE this.
    """
    x = particles[:, 0]
    y = particles[:, 1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    z = term1 + term2 + a + np.exp(1)
    return z

def ackley_hill(particles):
    """
    The inverted Ackley function (a hill with ripples).
    We will PLOT this.
    """
    return -ackley_valley(particles)
# --- END MODIFICATION ---


# --- MODIFIED: Changed the objective function to Ackley ---
# The objective_function for the PSO is the one it MINIMIZES
objective_function = ackley_valley
# --- END MODIFICATION ---


# --- PSO Hyperparameters ---
NUM_PARTICLES = 300
NUM_ITERATIONS = 200
NUM_INFORMANTS = 5
PARTICLE_LENGTH = 2  # Our particle is just [x, y]

PSO_PARAMS = {
    'alpha': 0.729,   
    'beta': 1.494,    
    'gamma': 1.0,     
    'delta': 1.494,   
    'epsilon': 0.01    
}

def main():
    """
    Main function to run the PSO on the Ackley benchmark
    and generate a 3D animation.
    """
    
    # --- 2. SETUP AND RUN PSO ---

    # Create initial random particles
    initial_particles = np.random.uniform(
        low=BOUND_LOW, 
        high=BOUND_HIGH, 
        size=(NUM_PARTICLES, PARTICLE_LENGTH)
    )

    # Initialize and Run Optimizer
    optimizer = pso.ParticleSwarm(
        num_particles=NUM_PARTICLES,
        num_informants=NUM_INFORMANTS,
        particle_length=PARTICLE_LENGTH,
        objective_function=objective_function,
        particles=initial_particles,
        num_iterations=NUM_ITERATIONS,
        **PSO_PARAMS
    )

    print("Starting PSO optimization (finding minimum of valley)...")
    for i in range(NUM_ITERATIONS):
        optimizer._update()
        if (i + 1) % 10 == 0:
            # The Gbest_value is the *valley's* minimum
            print(f"Iteration {i+1}/{NUM_ITERATIONS}, Best Valley 'Depth': {optimizer.Gbest_value:.6f}")

    print("\nOptimization Finished.")
    print(f"Best Position Found (x, y): {optimizer.Gbest}")

    # --- 3. SETUP THE VISUALIZATION ---

    particle_history = np.array(optimizer.particle_history)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # --- MODIFIED: Update titles ---
    ax.set_title("PSO climbing Ackley 'hill'")

    # Create the landscape (the "problem")
    X_plot = np.linspace(BOUND_LOW, BOUND_HIGH, 100)
    Y_plot = np.linspace(BOUND_LOW, BOUND_HIGH, 100)
    X_mesh, Y_mesh = np.meshgrid(X_plot, Y_plot)

    # --- MODIFIED: Plot the Ackley HILL function ---
    Z_mesh = ackley_hill(np.array([X_mesh.ravel(), Y_mesh.ravel()]).T).reshape(X_mesh.shape)

    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.6, rstride=1, cstride=1, edgecolor='none')

    # Create the particle scatter plot
    initial_pos = particle_history[0]
    # --- MODIFIED: Get particle height on the Ackley HILL ---
    particle_z = ackley_hill(initial_pos)
    scatter = ax.scatter(initial_pos[:, 0], initial_pos[:, 1], particle_z, 
                         color='red', s=40, depthshade=True, label="Particles")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y) - Fitness (Climbing)')
    ax.legend()

    # --- 4. CREATE THE ANIMATION ---
    # We define the update function inside main() so it has access to 
    # variables like 'particle_history', 'scatter', 'ax', etc.
    def update_animation(frame):
        """
        This function is called for each frame of the animation.
        """
        positions = particle_history[frame]
        
        # --- MODIFIED: Get particle height on the Ackley HILL ---
        particle_z = ackley_hill(positions)
        
        scatter._offsets3d = (positions[:, 0], positions[:, 1], particle_z)
        ax.set_title(f"Iteration {frame}/{NUM_ITERATIONS}")
        return scatter,

    ani = FuncAnimation(
        fig, 
        update_animation, 
        frames=NUM_ITERATIONS + 1, # +1 to include the last frame
        interval=100,
        blit=False
    )

    # --- 5. SAVE THE ANIMATION (Replaces plt.show()) ---

    def on_progress(current_frame, total_frames):
        """A helper function to show progress in the console."""
        percent = (current_frame / total_frames) * 100
        # Fix for large frame counts
        current_frame_str = f"Saving frame {current_frame} of {total_frames} ({percent:.1f}%)"
        sys.stdout.write(f'\r{current_frame_str:<80}') # Pad to overwrite previous line
        sys.stdout.flush()
        if current_frame == total_frames - 1:
            print("\nSaving complete!")

    try:
        print("Saving animation as pso_animation.gif...")
        # Warning: 1000 frames with 1000 particles will be slow and create a large file
        print(f"This may take a long time ({NUM_ITERATIONS+1} frames)...")
        ani.save(
            'pso_animation1.gif', 
            writer='pillow', 
            dpi=150, # Set a good resolution
            progress_callback=on_progress
        )
        # --- MODIFIED: Fixed the final print message ---
        print(f"\nSuccessfully saved animation to pso_animation.gif")

    except Exception as e:
        print(f"\nERROR saving animation: {e}")
        print("If you see 'unknown file extension: .mp4', you are using an old script.")
        print("Showing live animation instead:")
        plt.show()


# Standard Python entry point:
# This ensures that main() is only called when the script is
# executed directly (not when imported as a module).
if __name__ == "__main__":
    main()