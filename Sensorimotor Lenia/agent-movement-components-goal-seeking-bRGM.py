import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from ipywidgets import FloatSlider, Output, Button, Dropdown, HBox

# ========== bRGM Controller (Simplified) ============ (Copied from cell uKINjlcD_sUo)

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

# ... (Keep all previous class definitions unchanged)

class EnhancedBurstyBRGM(BurstyBRGM):
    def __init__(self, goal_strength=0.03, brm_strength=0.97):
        super().__init__()
        self.goal_strength = goal_strength
        self.brm_strength = brm_strength
        self.primary_velocity = np.array([0.,0.]) # Initialize to avoid NoneType errors
        self.brm_output = np.array([0.,0.]) # Initialize to avoid NoneType errors
        self.motor_command = np.array([0.,0.]) # Initialize to avoid NoneType errors
        self.agent_pos = np.array([-9.0, 0.0]) # Example initial value, will be updated in main
        self.goal_dist = np.linalg.norm(self.agent_pos - GOAL_POSITION) # Example initial value


    def step(self, sensor_values):
        # This step is for the animation loop to update internal state
        wall_dists_list, goal_dist = sensor_values
        # Store sensor values for potential use in update_components or for debugging
        self.wall_dists_list = wall_dists_list
        self.goal_dist = goal_dist

        # Sensor input for the underlying bRGM: wall cue and goal cue. Normalize them to a reasonable range for the bRGM.
        # Wall_dist max is ARENA_RADIUS * 2, goal_dist max is ARENA_RADIUS * 2.
        sensor_input_brgm = [min(v / (ARENA_RADIUS * 2), 1.0) for v in wall_dists_list] # Normalize wall distances
        sensor_input_brgm.append(min(goal_dist / (ARENA_RADIUS * 2), 1.0)) # Add normalized goal distance

        # Use a simplified combined input for the bRGM. The BurstyBRGM class expects 2 sensor inputs.
        # Let's average wall distances for S1 and use goal_dist for S2
        s1_brgm = np.mean(wall_dists_list) / (ARENA_RADIUS * 2)
        s2_brgm = goal_dist / (ARENA_RADIUS * 2)
        self.brgm_internal_output = super().step([s1_brgm, s2_brgm]) # Call parent's step method

    def update_components(self, agent_pos_current, goal_position_current):
        # Update internal agent position and goal position for component calculation
        self.agent_pos = agent_pos_current
        self.goal_position = goal_position_current

        # Calculate components for visualization
        self.primary_velocity = np.array([0., 0.])
        # Recalculate goal_dist based on current agent_pos
        current_goal_dist = np.linalg.norm(self.agent_pos - self.goal_position)

        if current_goal_dist > GOAL_RADIUS + AGENT_RADIUS:
            direction_vector = self.goal_position - self.agent_pos
            norm_direction = np.linalg.norm(direction_vector)
            if norm_direction > 0.001: # Avoid division by zero
                self.primary_velocity = direction_vector / norm_direction * self.goal_strength
            else:
                self.primary_velocity = np.array([0., 0.])

        self.brm_output = self.brgm_internal_output * self.brm_strength
        combined_motor_command = self.primary_velocity + self.brm_output

        norm_combined_motor_command = np.linalg.norm(combined_motor_command)
        if norm_combined_motor_command > 0.001:
            self.motor_command = combined_motor_command / norm_combined_motor_command # Normalize final movement direction
        else:
            self.motor_command = np.array([0., 0.])


# ================== Arena and Agent ================== (Copied from cell uKINjlcD_sUo)

ARENA_RADIUS = 12.0
GOAL_RADIUS = 0.5
GOAL_POSITION = np.array([9.0, 0.0])  # Near right wall
AGENT_RADIUS = 0.3
NUM_STEPS = 300

def arena_boundaries(pos):
    """Clamps agent position within the circular arena."""
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
    dist_s = y - (-ARENA_RADIUS) # Distance from y to bottom edge -ARENA_RADIUS
    dist_e = ARENA_RADIUS - x
    dist_w = x - (-ARENA_RADIUS) # Distance from x to left edge -ARENA_RADIUS

    wall_dists_list = [dist_n, dist_s, dist_e, dist_w]

    goal_dist = np.linalg.norm(agent_pos - GOAL_POSITION)
    return wall_dists_list, goal_dist


def main():
    # Initialize
    agent_pos = np.array([-9.0, 0.0])  # Start at left side
    brgm_controller = EnhancedBurstyBRGM(goal_strength=0.03, brm_strength=0.97)
    memory = [agent_pos.copy()]

    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    ax[0].set_title("Agent Movement with Component Visualization")
    ax[1].set_title("Primary Goal-Seeking Velocity")
    ax[2].set_title("bRGM Output Contribution")

    arena = plt.Circle((0,0), ARENA_RADIUS, color='gray', fill=False, linewidth=2)
    ax[0].add_patch(arena)
    goal_patch = plt.Circle(GOAL_POSITION, GOAL_RADIUS, color='green', fill=True, ec='black')
    ax[0].add_patch(goal_patch)
    ax[0].set_aspect('equal')
    ax[0].set_xlim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax[0].set_ylim(-ARENA_RADIUS-1, ARENA_RADIUS+1)

    agent_scatter = ax[0].scatter(agent_pos[0], agent_pos[1], c='red', s=60)
    trajectory_line, = ax[0].plot([], [], c='blue', lw=1.5)

    goal_text_artist = ax[0].text(GOAL_POSITION[0], GOAL_POSITION[1], 'Goal!', color='green', fontsize=14, ha='center', visible=False)

    # Initialize component plots
    time_points = []
    primary_vel_x = []
    primary_vel_y = []
    brm_x = []
    brm_y = []
    combined_x = []
    combined_y = []

    def init():
        nonlocal time_points, primary_vel_x, primary_vel_y, brm_x, brm_y, combined_x, combined_y
        time_points = []
        primary_vel_x = []
        primary_vel_y = []
        brm_x = []
        brm_y = []
        combined_x = []
        combined_y = []

        # Need to return all artists that animate
        return agent_scatter, trajectory_line, goal_patch, goal_text_artist, ax[1].plot([],[], 'r-')[0], ax[2].plot([],[], 'b-')[0]


    def update(frame):
        nonlocal agent_pos, memory

        wall_dists_list, goal_dist = sense_agent(agent_pos)
        brgm_controller.step((wall_dists_list, goal_dist))
        brgm_controller.update_components(agent_pos, GOAL_POSITION)

        # Update visualization data
        time_points.append(frame)
        primary_vel_x.append(brgm_controller.primary_velocity[0])
        primary_vel_y.append(brgm_controller.primary_velocity[1])
        brm_x.append(brgm_controller.brm_output[0])
        brm_y.append(brgm_controller.brm_output[1])
        combined_x.append(brgm_controller.motor_command[0])
        combined_y.append(brgm_controller.motor_command[1])

        # Update agent position
        agent_pos += brgm_controller.motor_command * 0.5 # Scale down movement speed
        agent_pos = arena_boundaries(agent_pos)
        memory.append(agent_pos.copy())

        # Update plots for main agent movement
        agent_scatter.set_offsets(agent_pos)
        trajectory_line.set_data(*zip(*memory))

        # Update plots for components
        ax[1].clear()
        ax[1].plot(time_points, primary_vel_x, 'r-', label='Primary X')
        ax[1].plot(time_points, primary_vel_y, 'r--', label='Primary Y')
        ax[1].legend()
        ax[1].set_title("Primary Goal-Seeking Velocity Components")
        ax[1].set_xlim(0, NUM_STEPS)
        ax[1].set_ylim(-1.0, 1.0) # Assuming normalized velocity components

        ax[2].clear()
        ax[2].plot(time_points, brm_x, 'b-', label='bRGM X')
        ax[2].plot(time_points, brm_y, 'b--', label='bRGM Y')
        ax[2].legend()
        ax[2].set_title("bRGM Output Contribution Components")
        ax[2].set_xlim(0, NUM_STEPS)
        ax[2].set_ylim(-1.0, 1.0) # Assuming normalized bRGM output components

        # Check for goal
        if np.linalg.norm(agent_pos - GOAL_POSITION) < GOAL_RADIUS + AGENT_RADIUS:
            goal_text_artist.set_visible(True)
            # Stop animation if goal is reached
            # ani.event_source.stop() # This will be set globally
        else:
            goal_text_artist.set_visible(False)

        # Return all artists that have been modified
        return agent_scatter, trajectory_line, goal_patch, goal_text_artist, ax[1].lines[0], ax[1].lines[1], ax[2].lines[0], ax[2].lines[1]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=NUM_STEPS, interval=80, blit=True
    )

    # Create individual sliders for Goal Strength and bRGM Strength
    goal_strength_slider = FloatSlider(
        value=0.5,
        min=0.0, max=1.0, step=0.1,
        description='Goal Strength:',
        orientation='horizontal',
        readout=True, readout_format='.1f',
        continuous_update=False
    )

    brm_strength_slider = FloatSlider(
        value=0.2,
        min=0.0, max=1.0, step=0.1,
        description='bRGM Strength:',
        orientation='horizontal',
        readout=True, readout_format='.1f',
        continuous_update=False
    )

    # Define options for the Dropdown
    mode_options = ['Combined', 'Pure Goal-Seeking', 'Pure bRGM']

    # Add interactive controls using the new sliders and corrected Dropdown
    controls = HBox([
        goal_strength_slider,
        brm_strength_slider,
        Button(description='Toggle bRGM Only', on_click=lambda b: toggle_brm_only()),
        Dropdown(options=mode_options, value=mode_options[0])
    ])

    def toggle_brm_only():
        # This function should update the brgm_controller's strengths
        if brgm_controller.goal_strength > 0 and brgm_controller.brm_strength > 0:
            brgm_controller.goal_strength = 0.0
            brgm_controller.brm_strength = 1.0
            print("Switched to bRGM Only mode.")
        else:
            brgm_controller.goal_strength = 0.03 # Default goal strength
            brgm_controller.brm_strength = 0.97 # Default bRGM strength
            print("Switched to Combined mode.")
        # You might need to re-run the animation to see the effect immediately

    # Connect sliders to update the brgm_controller's strengths
    def update_goal_strength(change):
        brgm_controller.goal_strength = change.new
        print(f"Goal Strength updated to {brgm_controller.goal_strength}")

    def update_brm_strength(change):
        brgm_controller.brm_strength = change.new
        print(f"bRGM Strength updated to {brgm_controller.brm_strength}")

    goal_strength_slider.observe(update_goal_strength, names='value')
    brm_strength_slider.observe(update_brm_strength, names='value')

    # Display everything
    display(controls)
    return HTML(ani.to_jshtml())
