import matplotlib.pyplot as plt
import numpy as np

# Increase steps to 200 for long-term analysis
steps = 200
soph_nav = SophisticatedTemporalAgent(state_dim=16)

hidden_state_history = []
mean_activations = []

# Run simulation
for i in range(steps):
    soph_nav.navigate_circuitously()
    hidden_state_history.append(soph_nav.hidden_state.copy())
    mean_activations.append(np.mean(soph_nav.hidden_state))

# Convert history to a numpy array for visualization
history_array = np.array(hidden_state_history)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

# Plot 1: Mean Activation over time
ax1.plot(mean_activations, color='#2ca02c', linewidth=1.5, label='Mean Hidden Activation')
ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.set_title(f'Long-term Evolution of Internal Memory ({steps} Steps)')
ax1.set_ylabel('Activation Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Heatmap of all hidden units
im = ax2.imshow(history_array.T, aspect='auto', cmap='RdYlBu', interpolation='nearest')
ax2.set_title('Hidden State Vector Heatmap (16 Dimensions)')
ax2.set_xlabel('Navigation Step')
ax2.set_ylabel('Hidden Unit Index')
fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15, label='Activation Level')

plt.tight_layout()
plt.show()
