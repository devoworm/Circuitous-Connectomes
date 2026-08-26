import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display # Added HTML, display for Colab animations

class BurstyBRGM50:
    def __init__(self):
        self.G = nx.DiGraph()
        # Five sensor nodes: S0 (North wall), S1 (South wall), S2 (East wall), S3 (West wall), S4 (Goal distance)
        self.sensor_nodes = [f"S{i}" for i in range(5)]
        self.intermediate_nodes = [f"I{i}" for i in range(50)]
        self.motor_x = "Mx"
        self.motor_y = "My"

        # Add nodes
        for n in self.sensor_nodes + self.intermediate_nodes + [self.motor_x, self.motor_y]:
            self.G.add_node(n)

        # Build path from sensors through intermediates to motors
        nodes = self.sensor_nodes + self.intermediate_nodes + [self.motor_x, self.motor_y]
        for i in range(len(nodes)-1):
            self.G.add_edge(nodes[i], nodes[i+1], weight=random.uniform(0.3, 0.8))

        # Add spaghettification (more cross-connections for bigger bRGM)
        for _ in range(200):
            u = random.choice(self.intermediate_nodes + self.sensor_nodes)
            v = random.choice(self.intermediate_nodes + [self.motor_x, self.motor_y])
            if u != v and not self.G.has_edge(u, v):
                self.G.add_edge(u, v, weight=random.uniform(0.05, 0.3))

        self.node_potentials = {n: 0.0 for n in self.G.nodes()}
        self.threshold = 0.5 # Lowered threshold to allow signals to propagate more easily
        self.burst_history = []

    def step(self, sensor_values):
        # sensor_values is now a tuple: (wall_distances_list, goal_distance_scalar)
        wall_dists_list, goal_dist_scalar = sensor_values

        # Normalizing sensor inputs to be between 0 and 1 before feeding to BRGM
        # The max possible wall_dist (single direction) or goal_dist is ARENA_RADIUS*2
        for i, s_val in enumerate(wall_dists_list):
            self.node_potentials[self.sensor_nodes[i]] = min(max(s_val / (ARENA_RADIUS * 2), 0.0), 1.0)
        self.node_potentials[self.sensor_nodes[4]] = min(max(goal_dist_scalar / (ARENA_RADIUS * 2), 0.0), 1.0)

        for n in self.intermediate_nodes + [self.motor_x, self.motor_y]:
            self.node_potentials[n] *= 0.7

        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        for node in nodes:
            if self.node_potentials[node] >= self.threshold:
                for nbr in list(self.G.successors(node)):
                    w = self.G[node][nbr]['weight']
                    self.node_potentials[nbr] += self.node_potentials[node] * w * 0.6
                self.node_potentials[node] = 0.0

        mx = self.node_potentials[self.motor_x]
        my = self.node_potentials[self.motor_y]
        if mx == 0 and my == 0:
            mx, my = np.random.randn(2) * 0.1 # Small random movement if no bursts
        return np.array([mx, my])

# === Arena and Agent ===

ARENA_RADIUS = 12.0
GOAL_RADIUS = 0.5
GOAL_POSITION = np.array([9.0, 0.0])
AGENT_RADIUS = 0.3
NUM_STEPS = 300  # Increased steps for longer simulation

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
      - wall_distances_list: [dist_n, dist_s, dist_e, dist_w]
      - dist_to_goal: distance to hidden goal
    """
    x, y = agent_pos
    dist_n = ARENA_RADIUS - y
    dist_s = y - (-ARENA_RADIUS)
    dist_e = ARENA_RADIUS - x
    dist_w = x - (-ARENA_RADIUS)

    wall_distances_list = [dist_n, dist_s, dist_e, dist_w]

    goal_dist = np.linalg.norm(agent_pos - GOAL_POSITION)
    return wall_distances_list, goal_dist

def main():
    agent_pos = np.array([-9.0, 0.0])
    brgm = BurstyBRGM50()
    memory = [agent_pos.copy()]

    fig, ax = plt.subplots()
    arena = plt.Circle((0,0), ARENA_RADIUS, color='gray', fill=False, linewidth=2)
    ax.add_patch(arena)
    goal_patch = plt.Circle(GOAL_POSITION, GOAL_RADIUS, color='green', fill=True, ec='black')
    ax.add_patch(goal_patch)
    ax.set_aspect('equal')
    ax.set_xlim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_ylim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_title("Sensorimotor Lenia with bRGM (50 intermediates)")

    agent_scatter = ax.scatter(agent_pos[0], agent_pos[1], c='red', s=60)
    trajectory_line, = ax.plot([], [], c='blue', lw=1.5)

    # Initialize text artist for 'Goal!' once, set initially invisible
    goal_text_artist = ax.text(GOAL_POSITION[0], GOAL_POSITION[1], 'Goal!', color='green', fontsize=14, ha='center', visible=False)

    # Make ani global to be able to stop it from within the update function
    global ani

    def update(frame):
        nonlocal agent_pos, memory
        wall_dists_list, goal_dist = sense_agent(agent_pos)

        # Step 1: Determine a primary goal-seeking velocity
        primary_velocity = np.array([0., 0.])
        if goal_dist > GOAL_RADIUS + AGENT_RADIUS: # Only seek goal if not already there
            direction_vector = GOAL_POSITION - agent_pos
            norm_direction = np.linalg.norm(direction_vector)
            if norm_direction > 0.01: # Avoid division by zero when very close
                primary_velocity = direction_vector / norm_direction # Normalized direction

        # Step 2: Get modulation from the bRGM
        # Sensor inputs for bRGM: wall_dists_list (list of 4 scalars) and goal_dist (scalar)
        brgm_sensor_input = (wall_dists_list, goal_dist) # Pass as tuple (list, scalar)
        brgm_output_raw = brgm.step(brgm_sensor_input) # This is [mx, my] potentials

        # Step 3: Combine primary velocity with bRGM output
        # We blend the primary goal-seeking direction with the bRGM's output.
        # This makes the agent primarily goal-seeking, with the bRGM adding modulation.
        goal_seeking_strength = 0.1 # How strongly it moves towards the goal
        brgm_influence_strength = 0.9 # How much the bRGM output perturbs the movement

        motor_command = primary_velocity * goal_seeking_strength + brgm_output_raw * brgm_influence_strength

        # Normalize the final motor command to control overall speed, unless it's very small
        command_norm = np.linalg.norm(motor_command)
        if command_norm > 0.01: # Ensure a minimum movement magnitude if not at rest
            final_delta = (motor_command / command_norm) * 0.5 # Overall movement speed, e.g., 1.0 unit per frame
        else:
            final_delta = np.array([0., 0.]) # Stay still

        agent_pos += final_delta
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

    ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS, interval=200, blit=True)
    plt.close(fig) # Prevent the static plot from showing
    return HTML(ani.to_jshtml()) # Display the animation in Colab

if __name__ == "__main__":
    display(main()) # Call main and display its return value (the HTML object)
