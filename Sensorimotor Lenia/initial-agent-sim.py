## Recreating Sensorimotor Lenia using OLMO 3.1 32b Instruct

import numpy as np

# Grid size
GRID_SIZE = 50
# Each cell: [alive, excitation]
cells = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=float)  # alive, excitation

# Randomly initialize some cells
for _ in range(500): # Increased the number of initially alive cells
    x, y = np.random.randint(0, GRID_SIZE, 2)
    cells[x, y, 0] = 1.0  # alive
    cells[x, y, 1] = 0.95  # Increased initial excitation further for more persistence

import ipywidgets as widgets
from IPython.display import display, HTML

# Initial definition of ALIVE_STATUS_THRESHOLD as a global variable
ALIVE_STATUS_THRESHOLD = 0.3

def step():
    global cells  # Declare 'cells' as global
    # Copy grid to avoid in-place updates
    new_cells = cells.copy()

    # Neighborhood kernel (Moore, 8 neighbors)
    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            alive, excitation = cells[x, y]

            if alive > 0.6:  # if cell is truly alive (after potential death rule)
                # Sense neighbors: sum of neighbors' alive states (toroidal)
                neighbor_alive_sum = 0.0
                for dx, dy in neighbors:
                    nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE # Toroidal wrapping
                    neighbor_alive_sum += cells[nx, ny, 0]

                # If excitation is high, try to move with a more complex heuristic
                if excitation > 0.4:  # Lowered movement excitation threshold from 0.5 to 0.4
                    candidate_moves = [] # Store (score, dx, dy)

                    # Shuffle neighbors to remove directional bias for tie-breaking
                    shuffled_neighbors = list(neighbors)
                    np.random.shuffle(shuffled_neighbors)

                    for dx, dy in shuffled_neighbors:
                        nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE # Toroidal wrapping
                        neighbor_crowding = cells[nx, ny, 0] # Sense previous state
                        neighbor_excitation = cells[nx, ny, 1] # Sense previous state

                        # Heuristic: Minimize score. Prioritize empty spots (low crowding),
                        # then among empty, prioritize spots with higher excitation.
                        CROWDING_PENALTY_WEIGHT = 100.0 # High penalty for existing agents
                        EXCITATION_ATTRACTION_BONUS = 1.0 # Bonus for high excitation

                        # Score: (crowding * weight) - (excitation * bonus)
                        # Empty cells will have scores from -EXCITATION_ATTRACTION_BONUS to 0.
                        # Crowded cells will have high positive scores.
                        move_score = (neighbor_crowding * CROWDING_PENALTY_WEIGHT) - (neighbor_excitation * EXCITATION_ATTRACTION_BONUS)
                        candidate_moves.append((move_score, dx, dy))

                    if candidate_moves:
                        # Sort candidates by score (ascending) to find the best move
                        candidate_moves.sort(key=lambda item: item[0])
                        best_move_score, best_dx, best_dy = candidate_moves[0]

                        target_nx, target_ny = (x + best_dx) % GRID_SIZE, (y + best_dy) % GRID_SIZE # Toroidal wrapping

                        # Only move if the chosen target spot is effectively empty
                        # (best_move_score should not indicate a crowded cell, and cell must be truly empty)
                        if best_move_score < CROWDING_PENALTY_WEIGHT / 2 and cells[target_nx, target_ny, 0] < 0.1:
                            # Remove from old position
                            new_cells[x, y, 0] = 0.0
                            new_cells[x, y, 1] = 0.0 # Old spot loses excitation

                            # Add to new position (alive status will be determined by final thresholding)
                            # new_cells[target_nx, target_ny, 0] = 1.0
                            # Maintain excitation at new spot (decay slightly, but allow continued movement)
                            new_cells[target_nx, target_ny, 1] = excitation * 0.99 # Increased retention

                            # Spread excitation from the *new* position to its neighbors (toroidal)
                            for dx2, dy2 in neighbors:
                                nnx, nny = (target_nx + dx2) % GRID_SIZE, (target_ny + dy2) % GRID_SIZE # Toroidal wrapping
                                new_cells[nnx, nny, 1] += excitation * 0.2 # Increased spread for moving agent

                        else: # Agent stays at (x,y) - either no empty spot or excitation was not high enough for a good move
                            # Decay excitation but re-charge slightly if staying
                            new_cells[x, y, 1] = excitation * 0.9 + 0.2 # Reduced decay for staying agent, increased recharge
                            # Spread excitation from current position to its neighbors (toroidal)
                            for dx2, dy2 in neighbors:
                                nnx, nny = (x + dx2) % GRID_SIZE, (y + dy2) % GRID_SIZE # Toroidal wrapping
                                new_cells[nnx, nny, 1] += excitation * 0.1 # Spread for staying agent

                else: # excitation too low to attempt a complex move (threshold 0.4 was not met)
                    # No move, re-charge slightly
                    new_cells[x, y, 1] = excitation * 0.8 + 0.3 # Reduced decay for low excitation agent, adjusted recharge
                    # Spread excitation from current position to its neighbors (toroidal)
                    for dx2, dy2 in neighbors:
                        nnx, nny = (x + dx2) % GRID_SIZE, (y + dy2) % GRID_SIZE # Toroidal wrapping
                        new_cells[nnx, nny, 1] += excitation * 0.1 # Spread for low excitation agent
            else: # cell is dead (alive <= 0.6) and not 'newly dead' from the threshold rule
                # Dead cell: excitation decays quickly but does not reach 0 (persistence floor)
                new_cells[x, y, 1] = max(0.1, excitation * 0.5) # Ensure excitation has a higher floor (0.1) and slower decay

    # Clip excitation values and ensure a minimum floor for all excitation (persistence)
    new_cells[:, :, 1] = np.clip(new_cells[:, :, 1], 0.1, 1) # Excitation floor at 0.1

    ## Refinement for 1:1 mapping: Update alive status based on excitation
    ## The 'alive' status (cells[:,:,0]) is now a direct thresholding of the excitation (cells[:,:,1]).
    ## This ensures a strong 1:1 visual correlation and consistent logic with 'alive > 0.6' checks.
    ALIVE_STATUS_THRESHOLD = 0.3 # Lowered to increase visual persistence
    new_cells[:, :, 0] = (new_cells[:, :, 1] >= ALIVE_STATUS_THRESHOLD).astype(float)
    cells = new_cells

def interactive_simulation_control(threshold):
    global ALIVE_STATUS_THRESHOLD
    ALIVE_STATUS_THRESHOLD = threshold
    print(f"ALIVE_STATUS_THRESHOLD set to {ALIVE_STATUS_THRESHOLD:.2f}.")
    print("Please re-run Cell `ReurRgLqndyq` to re-initialize the grid, then re-run the animation cells (`Tm50549monIc` and `i6DEEQPCnzNq`) to observe the changes.")

alive_threshold_slider = widgets.FloatSlider(
    value=0.3,
    min=0.0,
    max=1.0,
    step=0.05,
    description='Alive Threshold:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
)

print("Use the slider below to adjust the ALIVE_STATUS_THRESHOLD.")
print("After adjusting, remember to re-run the initialization and animation cells.")

interact_widget = widgets.interactive(interactive_simulation_control, threshold=alive_threshold_slider)
display(interact_widget)

## Visualization animation (by alive status)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

fig, ax = plt.subplots()
im = ax.imshow(cells[:,:,0], cmap='gray', vmin=0, vmax=1)

def update(frame):
    step()
    im.set_data(cells[:,:,0])
    return im,

ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
plt.close(fig) # Prevent the static plot from showing
HTML(ani.to_jshtml())
