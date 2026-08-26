## bRGM with 5 intermediate nodes

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display # Added HTML, display for Colab animations

class BurstyBRGM5:
    def __init__(self):
        self.G = nx.DiGraph()
        self.sensor1 = "S1"
        self.sensor2 = "S2"
        self.intermediate_nodes = [f"I{i}" for i in range(5)]
        self.motor_x = "Mx"
        self.motor_y = "My"

        # Add nodes
        for n in [self.sensor1, self.sensor2] + self.intermediate_nodes + [self.motor_x, self.motor_y]:
            self.G.add_node(n)

        # Build path from sensor to motor
        nodes = [self.sensor1] + self.intermediate_nodes + [self.motor_x, self.motor_y]
        for i in range(len(nodes)-1):
            self.G.add_edge(nodes[i], nodes[i+1], weight=random.uniform(0.3, 0.8))

        # Add spaghettification (random cross-connections)
        for _ in range(10):
            u = random.choice(self.intermediate_nodes + [self.sensor1, self.sensor2])
            v = random.choice(self.intermediate_nodes + [self.motor_x, self.motor_y])
            if u != v and not self.G.has_edge(u, v):
                self.G.add_edge(u, v, weight=random.uniform(0.05, 0.3))

        self.node_potentials = {n: 0.0 for n in self.G.nodes()}
        self.threshold = 1.0
        self.burst_history = []

    def step(self, sensor_values):
        s1, s2 = sensor_values
        # Normalize sensor values
        self.node_potentials[self.sensor1] = min(max(s1 / 12.0, 0.0), 1.0)
        self.node_potentials[self.sensor2] = min(max(s2 / 12.0, 0.0), 1.0)
        # Decay intermediates and motors
        for n in self.intermediate_nodes + [self.motor_x, self.motor_y]:
            self.node_potentials[n] *= 0.7

        # Propagate in random order
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
            mx, my = np.random.randn(2) * 0.1
        return np.array([mx, my])

# === Arena and Agent ===

ARENA_RADIUS = 12.0
GOAL_RADIUS = 0.5
GOAL_POSITION = np.array([9.0, 0.0])
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
    agent_pos = np.array([-9.0, 0.0])
    brgm = BurstyBRGM5()
    memory = [agent_pos.copy()]

    fig, ax = plt.subplots()
    arena = plt.Circle((0,0), ARENA_RADIUS, color='gray', fill=False, linewidth=2)
    ax.add_patch(arena)
    goal_patch = plt.Circle(GOAL_POSITION, GOAL_RADIUS, color='green', fill=True, ec='black')
    ax.add_patch(goal_patch)
    ax.set_aspect('equal')
    ax.set_xlim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_ylim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_title("Sensorimotor Lenia with bRGM (5 intermediates)")

    agent_scatter = ax.scatter(agent_pos[0], agent_pos[1], c='red', s=60)
    trajectory_line, = ax.plot([], [], c='blue', lw=1.5)

    # Initialize text artist for 'Goal!' once, set initially invisible
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
    return HTML(ani.to_jshtml()) # Display the animation in Colab

if __name__ == "__main__":
    display(main()) # Call main and display its return value (the HTML object)
