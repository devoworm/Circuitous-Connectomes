import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Improved canalization function
def canalization_function(brgm_size, initial_threshold=1.0, scale_factor=0.1, shift=50.0,
                          min_threshold=0.1, max_threshold=2.0):
    sigmoid_input = -scale_factor * (brgm_size - shift)
    sigmoid_output = 1 / (1 + np.exp(sigmoid_input))
    adjusted_threshold = initial_threshold + sigmoid_output
    return np.clip(adjusted_threshold, min_threshold, max_threshold)

# Network parameters
grid_size = 20  # 20x20 neuron grid
activation_levels = np.random.randn(grid_size, grid_size)  # Random "neural activity"

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Developmental Canalization in Neural Activation")
ax.axis('off')

# Create initial image (all neurons inactive)
im = ax.imshow(np.zeros((grid_size, grid_size)), cmap='viridis', vmin=0, vmax=1)
colorbar = fig.colorbar(im, ax=ax, shrink=0.5)
colorbar.set_label("Activation Level")

def update(frame):
    """Update activation map based on canalization threshold"""
    brgm_size = frame  # Vary network "complexity"
    threshold = canalization_function(brgm_size)

    # Neurons activate if activity exceeds threshold
    active_map = activation_levels > threshold
    active_map = active_map.astype(float)  # 0/1 → 0.0/1.0 for visualization

    # Update display
    im.set_data(active_map)
    ax.set_title(f"BRGM Size={brgm_size:.0f} | Threshold={threshold:.2f}")
    return im,

# Animation parameters
ani = FuncAnimation(fig, update,
                   frames=np.linspace(0, 100, 20),  # Smooth transition
                   interval=10,
                   blit=True)

# Save animation (or use plt.show() in notebooks)
ani.save("canalization_activation.mp4",
        writer="ffmpeg", fps=1,
        metadata=dict(artist="AI Assistant"))
print("Animation saved as 'canalization_activation.mp4'")

# To display in notebooks:
# plt.close()
# from IPython.display import HTML
# HTML(ani.to_jshtml())
