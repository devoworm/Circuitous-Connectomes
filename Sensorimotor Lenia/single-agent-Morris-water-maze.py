## A single sensorimotor Lenia agent in a Morris water maze/spatial cognition task.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML # Import HTML for Colab animation display

# Arena parameters
ARENA_RADIUS = 20.0
GOAL_RADIUS = 0.5
GOAL_ANGLE = 45.0 # Degrees, initial angle for goal placement (0 degrees = east)
# Calculate initial GOAL_POSITION based on GOAL_ANGLE
angle_rad = np.deg2rad(GOAL_ANGLE)
GOAL_POSITION = np.array([(ARENA_RADIUS - GOAL_RADIUS) * np.cos(angle_rad),
                          (ARENA_RADIUS - GOAL_RADIUS) * np.sin(angle_rad)])
GOAL_SPEED = 0.5 # How fast the goal moves
GOAL_VELOCITY = np.array([0.0, 0.0]) # Initial goal velocity
NUM_STEPS = 500

# Agent parameters
AGENT_RADIUS = 0.05
agent_pos = np.array([0.0, 0.0])  # Start at left side
agent_vel = np.zeros(2)
memory = []  # To keep track of past positions for visualization
EXPLORATION_STRENGTH = 0.95 # Increased exploration strength

def arena_boundaries(pos):
    """Clamps agent position within the circular arena."""
    current_dist = np.linalg.norm(pos)
    if current_dist > ARENA_RADIUS - AGENT_RADIUS:
        # Normalize the position vector and scale it back to the allowed radius
        pos_unit_vector = pos / current_dist
        pos = pos_unit_vector * (ARENA_RADIUS - AGENT_RADIUS)
    return pos

def goal_arena_boundaries(pos):
    """Clamps goal position within the circular arena."""
    current_dist = np.linalg.norm(pos)
    if current_dist > ARENA_RADIUS - GOAL_RADIUS:
        # Normalize the position vector and scale it back to the allowed radius
        pos_unit_vector = pos / current_dist
        pos = pos_unit_vector * (ARENA_RADIUS - GOAL_RADIUS)
    return pos

def sense_agent(agent_pos):
    """
    The agent senses:
      - Distance to the nearest wall
      - Distance to the goal (but not directly observable; we'll use a proxy)
    For this demo, agent can sense:
      - Distance to wall (as a vector of distances in 4 directions: N, S, E, W)
      - Distance to goal (used internally by the controller, not by the agent directly)
    """
    # Distances to walls in 4 cardinal directions
    x, y = agent_pos
    # Distance from center for radial boundary
    dist_from_center = np.linalg.norm(agent_pos)

    # Distances to cardinal 'walls' based on current position and arena boundaries
    dist_n = ARENA_RADIUS - y
    dist_s = y - (-ARENA_RADIUS)
    dist_e = ARENA_RADIUS - x
    dist_w = x - (-ARENA_RADIUS)

    wall_dist = np.array([dist_n, dist_s, dist_e, dist_w])
    # Normalize (max distance is 2*ARENA_RADIUS, min is 0)
    wall_dist = wall_dist / (2 * ARENA_RADIUS)

    # Distance to goal (hidden, but we can let the agent have access for demo purposes)
    goal_dist = np.linalg.norm(agent_pos - GOAL_POSITION)
    return wall_dist, goal_dist

def controller(agent_pos, wall_dist, goal_dist, memory):
    """
    Simple controller:
      - Bias movement toward the goal if close, else follow wall cues.
      - Add some randomness for exploration.
    """
    # If close to goal, move directly toward it
    if goal_dist < 1.0:
        # Ensure we don't divide by zero if goal_dist is extremely small
        if goal_dist == 0:
            direction = np.array([0., 0.])
        else:
            direction = (GOAL_POSITION - agent_pos) / goal_dist
    else:
        # If close to a wall, turn away
        if wall_dist[np.argmin(wall_dist)] < 0.2: # Using raw wall_dist, not normalized
            min_wall_idx = np.argmin(wall_dist)
            # Define directional vectors to move AWAY from each wall:
            # 0: North wall closest -> move South (0, -1)
            # 1: South wall closest -> move North (0, 1)
            # 2: East wall closest -> move West (-1, 0)
            # 3: West wall closest -> move East (1, 0)

            away_from_wall_vectors = np.array([
                [0., -1.], # Move South (away from North wall)
                [0.,  1.], # Move North (away from South wall)
                [-1., 0.], # Move West (away from East wall)
                [1.,  0.]  # Move East (away from West wall)
            ], dtype=float)

            direction = away_from_wall_vectors[min_wall_idx]
        else:
            # Move in the direction toward the goal projected onto the arena
            # Ensure we don't divide by zero if agent_pos == GOAL_POSITION
            goal_vec_norm = np.linalg.norm(GOAL_POSITION - agent_pos)
            if goal_vec_norm == 0:
                direction = np.array([0., 0.])
            else:
                direction = (GOAL_POSITION - agent_pos) / goal_vec_norm

            # Project onto the arena (avoid going outside) - this logic is implicitly handled by arena_boundaries post-move
            # The original 'clamp to arena' logic here is redundant with arena_boundaries

    # Add exploration noise after primary direction is determined
    direction = np.array(direction, dtype=float) # Ensure float type
    direction += np.random.randn(2) * EXPLORATION_STRENGTH

    # Normalize direction if it's not a zero vector
    norm_direction = np.linalg.norm(direction)
    if norm_direction > 0:
        direction /= norm_direction
    else:
        direction = np.array([0., 0.]) # Ensure it remains a 2D array if magnitude is zero

    return direction

fig, ax = plt.subplots()
arena = plt.Circle((0,0), ARENA_RADIUS, color='gray', fill=False, linewidth=2)
ax.add_patch(arena)
goal_patch = plt.Circle(GOAL_POSITION, GOAL_RADIUS, color='green', fill=True, ec='black')
ax.add_patch(goal_patch)
ax.set_aspect('equal')
ax.set_xlim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
ax.set_ylim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
ax.set_title("Sensorimotor Lenia Agent in Morris Maze-like Arena")

agent_scatter = ax.scatter(agent_pos[0], agent_pos[1], c='red', s=40)
trajectory_line, = ax.plot([], [], c='blue', lw=1)
# Create a text artist for 'Goal!' once, set it initially invisible
goal_text_artist = ax.text(GOAL_POSITION[0], GOAL_POSITION[1], 'Goal!', color='green', fontsize=14, ha='center', visible=False)

# Initialize ani to None so it can be declared global in update
ani = None

def update(frame):
    global agent_pos, memory, ani, GOAL_POSITION, GOAL_VELOCITY, GOAL_SPEED, goal_text_artist

    # Update goal position
    if frame % 20 == 0: # Change goal direction every 20 frames
        angle = np.random.uniform(0, 2 * np.pi)
        GOAL_VELOCITY = np.array([np.cos(angle), np.sin(angle)]) * GOAL_SPEED

    GOAL_POSITION += GOAL_VELOCITY
    GOAL_POSITION = goal_arena_boundaries(GOAL_POSITION) # Ensure goal stays within arena
    goal_patch.set_center(GOAL_POSITION)
    goal_text_artist.set_position(GOAL_POSITION) # Update text position as well

    wall_dist, goal_dist = sense_agent(agent_pos)
    direction = controller(agent_pos, wall_dist, goal_dist, memory)
    agent_pos += 0.2 * direction
    agent_pos = arena_boundaries(agent_pos)
    memory.append(agent_pos.copy())
    agent_scatter.set_offsets(agent_pos)
    trajectory_line.set_data(*zip(*memory))
    ax.set_autoscale_on(False)

    # Check if goal is reached
    if np.linalg.norm(agent_pos - GOAL_POSITION) < GOAL_RADIUS:
        goal_text_artist.set_visible(True)
        if ani: # Ensure ani is not None before trying to stop
            ani.event_source.stop() # Stop the animation
    else:
        goal_text_artist.set_visible(False) # Ensure text is hidden if not at goal

    # Always return all modified artists
    return agent_scatter, trajectory_line, goal_patch, goal_text_artist

ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS, interval=80, blit=True)
plt.close(fig) # Prevent the static plot from showing
HTML(ani.to_jshtml())

import matplotlib.pyplot as plt

# Increase the animation embed limit to avoid frames being dropped for larger animations
# The value is in MB. 100MB should be sufficient for this simulation with trajectories.
plt.rcParams['animation.embed_limit'] = 100.0
print(f"Matplotlib animation embed limit set to {plt.rcParams['animation.embed_limit']} MB.")

import ipywidgets as widgets
from IPython.display import display, HTML

def interactive_goal_angle_control(angle):
    global GOAL_ANGLE, GOAL_POSITION # Need to update both
    GOAL_ANGLE = angle
    # Recalculate GOAL_POSITION based on the new GOAL_ANGLE
    angle_rad = np.deg2rad(GOAL_ANGLE)
    GOAL_POSITION = np.array([(ARENA_RADIUS - GOAL_RADIUS) * np.cos(angle_rad),
                              (ARENA_RADIUS - GOAL_RADIUS) * np.sin(angle_rad)])
    print(f"GOAL_ANGLE set to {GOAL_ANGLE:.1f} degrees. Initial GOAL_POSITION: {GOAL_POSITION}.")
    print("Please re-run Cell `ojeSaohAhVux` to re-initialize the simulation with the new goal angle.")

goal_angle_slider = widgets.FloatSlider(
    value=GOAL_ANGLE, # Use the current GOAL_ANGLE from the simulation
    min=0.0,
    max=360.0,
    step=5.0,
    description='Goal Angle (degrees):',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

print("Use the slider below to adjust the initial angular position of the goal.")
print("After adjusting, remember to re-run Cell `ojeSaohAhVux` to observe the changes.")

interact_widget_angle = widgets.interactive(interactive_goal_angle_control, angle=goal_angle_slider)
display(interact_widget_angle)

## Hybrid Sensorimotor Lenia with bursty bRGM controller

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

# ========== bRGM Controller (Simplified) ============

class BurstyBRGM:
    def __init__(self, num_intermediate=8):
        """
        Sensor input: 2 values (e.g., distance to wall N/S/E/W or to goal)
        Output: 2 values (motor command for x, y)
        """
        self.G = nx.DiGraph()
        # Define sensor and motor nodes
        self.sensor1 = "S1"
        self.sensor2 = "S2"
        self.intermediate_nodes = [f"I{i}" for i in range(num_intermediate)]
        self.motor_x = "Mx"
        self.motor_y = "My"

        # Input nodes
        self.G.add_node(self.sensor1)
        self.G.add_node(self.sensor2)

        # Build a path from sensors through intermediates to motors
        nodes = [self.sensor1] + self.intermediate_nodes + [self.motor_x, self.motor_y]
        for i in range(len(nodes)-1):
            self.G.add_edge(nodes[i], nodes[i+1], weight=random.uniform(0.3, 0.8))

        # Add spaghettification (random cross-connections)
        for _ in range(num_intermediate*2):
            u = random.choice(self.intermediate_nodes + [self.sensor1, self.sensor2])
            v = random.choice(self.intermediate_nodes + [self.motor_x, self.motor_y])
            if u != v and not self.G.has_edge(u, v):
                self.G.add_edge(u, v, weight=random.uniform(0.05, 0.3))

        # Simulation state
        self.node_potentials = {n: 0.0 for n in self.G.nodes()}
        self.threshold = 1.0
        self.burst_history = []

    def step(self, sensor_values):
        """
        sensor_values: [s1, s2] (e.g., wall cue, goal cue)
        """
        # Inject input into sensor nodes (normalize for variety)
        s1, s2 = sensor_values
        # Normalize to [0,1] for variety
        self.node_potentials[self.sensor1] = min(max(s1 / 12.0, 0.0), 1.0)  # Arena radius ~12
        self.node_potentials[self.sensor2] = min(max(s2 / 12.0, 0.0), 1.0)
        # Reset intermediates and motors
        for n in self.intermediate_nodes + [self.motor_x, self.motor_y]:
            self.node_potentials[n] *= 0.7  # Decay

        # Propagate and check for bursts
        active_nodes = []
        # Process in random order to simulate asynchrony
        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        for node in nodes:
            if self.node_potentials[node] >= self.threshold:
                active_nodes.append(node)
                # Burst: propagate to neighbors
                for nbr in list(self.G.successors(node)):
                    w = self.G[node][nbr]['weight']
                    self.node_potentials[nbr] += self.node_potentials[node] * w * 0.6
                # Reset potential
                self.node_potentials[node] = 0.0
        # Read out motor outputs
        mx = self.node_potentials[self.motor_x]
        my = self.node_potentials[self.motor_y]
        # Nonlinearity: if nothing fired, small random movement
        if mx == 0 and my == 0:
            mx, my = np.random.randn(2) * 0.1
        return np.array([mx, my])

# ================== Arena and Agent ==================

ARENA_RADIUS = 12.0
GOAL_RADIUS = 0.5
GOAL_POSITION = np.array([9.0, 0.0])  # Near right wall
AGENT_RADIUS = 0.3
NUM_STEPS = 300

def arena_boundaries(pos):
    """Reflect agent if out of bounds."""
    # Ensure pos is an array of floats to avoid integer division issues
    pos_f = pos.astype(float)
    current_dist = np.linalg.norm(pos_f)
    if current_dist > ARENA_RADIUS - AGENT_RADIUS:
        # Normalize the position vector and scale it back to the allowed radius
        pos_unit_vector = pos_f / current_dist
        return pos_unit_vector * (ARENA_RADIUS - AGENT_RADIUS)
    return pos_f

def sense_agent(agent_pos):
    """
    Returns:
      - dist_to_wall: average distance to nearest wall (in 4 directions)
      - dist_to_goal: distance to hidden goal
    """
    x, y = agent_pos
    dist_n = ARENA_RADIUS - y
    dist_s = y - (-ARENA_RADIUS) # Distance from y to bottom edge -ARENA_RADIUS
    dist_e = ARENA_RADIUS - x
    dist_w = x - (-ARENA_RADIUS) # Distance from x to left edge -ARENA_RADIUS

    # Use the minimum distance to a wall, not average, to represent 'nearest'
    wall_dist = min(dist_n, dist_s, dist_e, dist_w)

    goal_dist = np.linalg.norm(agent_pos - GOAL_POSITION)
    return wall_dist, goal_dist

def main():
    # Initialize
    agent_pos = np.array([-9.0, 0.0])  # Start at left side
    brgm = BurstyBRGM(num_intermediate=8)
    memory = [agent_pos.copy()]

    fig, ax = plt.subplots()
    arena = plt.Circle((0,0), ARENA_RADIUS, color='gray', fill=False, linewidth=2)
    ax.add_patch(arena)
    goal_patch = plt.Circle(GOAL_POSITION, GOAL_RADIUS, color='green', fill=True, ec='black')
    ax.add_patch(goal_patch)
    ax.set_aspect('equal')
    ax.set_xlim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_ylim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_title("Sensorimotor Lenia with bRGM Controller")

    agent_scatter = ax.scatter(agent_pos[0], agent_pos[1], c='red', s=60)
    trajectory_line, = ax.plot([], [], c='blue', lw=1.5)

    # Initialize text artist for 'Goal!' once
    goal_text_artist = ax.text(GOAL_POSITION[0], GOAL_POSITION[1], 'Goal!', color='green', fontsize=14, ha='center', visible=False)

    # Make ani global to be able to stop it from within the update function
    global ani

    def update(frame):
        nonlocal agent_pos, memory
        wall_dist, goal_dist = sense_agent(agent_pos)
        # Sensor input: wall cue and goal cue. Normalize them to a reasonable range for the bRGM.
        # Wall_dist max is ARENA_RADIUS * 2, goal_dist max is ARENA_RADIUS * 2.
        sensor_input = [wall_dist / (ARENA_RADIUS * 2), goal_dist / (ARENA_RADIUS * 2)]
        motor_delta = brgm.step(sensor_input) * 0.5 # Scale down motor output for smoother movement
        agent_pos += motor_delta
        agent_pos = arena_boundaries(agent_pos)
        memory.append(agent_pos.copy())
        agent_scatter.set_offsets(agent_pos)
        trajectory_line.set_data(*zip(*memory))

        # Check for goal
        if np.linalg.norm(agent_pos - GOAL_POSITION) < GOAL_RADIUS + AGENT_RADIUS:
            goal_text_artist.set_visible(True)
            # Stop animation if goal is reached
            if ani is not None:
                ani.event_source.stop()
        else:
            goal_text_artist.set_visible(False)

        # The update function must return all artists that have been modified.
        return agent_scatter, trajectory_line, goal_patch, goal_text_artist

    ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS, interval=80, blit=True)
    plt.close(fig) # Prevent the static plot from showing
    return HTML(ani.to_jshtml())

if __name__ == "__main__":
    # Display the animation HTML object directly
    display(main())
