import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import random
import networkx as nx
from matplotlib.collections import LineCollection

# ==============================================================================
#                           Necessary Definitions from rsEDMO9G6BIP
# ==============================================================================

# Arena parameters
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

# ==============================================================================

NUM_AGENTS = 10 # Number of agents in the simulation

def main_multi_agent():
    # Initialize 100 agents
    agents_list = []
    for _ in range(NUM_AGENTS):
        # Random initial position within the arena (not too close to the edge)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, ARENA_RADIUS - AGENT_RADIUS - 1) # Ensure agents don't start at the very edge
        start_x = radius * np.cos(angle)
        start_y = radius * np.sin(angle)

        agents_list.append({
            'pos': np.array([start_x, start_y]),
            'controller': EnhancedBurstyBRGM(goal_strength=0.03, brm_strength=0.97), # Using default strengths
            'trajectory': [np.array([start_x, start_y])] # Store initial position
        })

    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    arena_patch = plt.Circle((0,0), ARENA_RADIUS, color='gray', fill=False, linewidth=2)
    ax.add_patch(arena_patch)
    goal_patch = plt.Circle(GOAL_POSITION, GOAL_RADIUS, color='green', fill=True, ec='black')
    ax.add_patch(goal_patch)
    ax.set_aspect('equal')
    ax.set_xlim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_ylim(-ARENA_RADIUS-1, ARENA_RADIUS+1)
    ax.set_title(f"Multi-Agent Sensorimotor Lenia ({NUM_AGENTS} Agents)")

    # Initial scatter plot for all agents' current positions
    agent_positions_data = np.array([agent['pos'] for agent in agents_list])
    agents_scatter = ax.scatter(agent_positions_data[:,0], agent_positions_data[:,1], c='red', s=20)

    # Create LineCollection objects for each agent's trajectory
    trajectory_collections = []
    for _ in range(NUM_AGENTS):
        lc = LineCollection([], linewidths=0.5, alpha=0.7)
        ax.add_collection(lc)
        trajectory_collections.append(lc)

    goal_text_artist = ax.text(GOAL_POSITION[0], GOAL_POSITION[1], 'Goal!', color='green', fontsize=14, ha='center', visible=False)

    TRAJECTORY_LENGTH = 50 # How many past steps to show for trajectory

    # Define a colormap for trajectory fading
    cmap = plt.colormaps['plasma'] # Use recommended way

    def update_multi_agent(frame):
        nonlocal agents_list

        current_agent_positions = []
        num_agents_at_goal = 0

        for i, agent in enumerate(agents_list):
            # Sense environment
            wall_dists_list, goal_dist = sense_agent(agent['pos'])

            # Update controller's internal state
            agent['controller'].step((wall_dists_list, goal_dist))
            agent['controller'].update_components(agent['pos'], GOAL_POSITION)

            # Apply movement
            agent['pos'] += agent['controller'].motor_command * 0.5 # Scale movement speed
            agent['pos'] = arena_boundaries(agent['pos']) # Ensure agent stays within arena

            # Update trajectory
            agent['trajectory'].append(agent['pos'].copy())
            if len(agent['trajectory']) > TRAJECTORY_LENGTH:
                agent['trajectory'].pop(0)

            current_agent_positions.append(agent['pos'])

            # Update individual trajectory line with color gradient
            path = np.array(agent['trajectory'])
            if len(path) > 1:
                points = path.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create colors for the gradient along the trajectory (from older to newer)
                colors = cmap(np.linspace(0, 1, len(segments)))

                trajectory_collections[i].set_segments(segments)
                trajectory_collections[i].set_colors(colors)
            else:
                trajectory_collections[i].set_segments([]) # Clear segments if not enough points

            # Check if agent reached goal
            if np.linalg.norm(agent['pos'] - GOAL_POSITION) < GOAL_RADIUS + AGENT_RADIUS:
                num_agents_at_goal += 1

        # Update the scatter plot with new current positions
        agents_scatter.set_offsets(np.array(current_agent_positions))

        # Show 'Goal!' text if any agent reached it
        if num_agents_at_goal > 0:
            goal_text_artist.set_visible(True)
        else:
            goal_text_artist.set_visible(False)

        # Return all modified artists
        return (agents_scatter, goal_text_artist, goal_patch) + tuple(trajectory_collections)

    # Create and return the animation
    ani = animation.FuncAnimation(fig, update_multi_agent, frames=NUM_STEPS, interval=80, blit=True)
    plt.close(fig) # Prevent the static plot from showing
    return ani # Return the animation object directly

if __name__ == '__main__':
    # The animation will be handled in a separate cell for saving purposes
    pass

def run_simulation_for_analysis(num_agents, num_steps, arena_radius, agent_radius, goal_radius, goal_position):
    """
    Runs the multi-agent simulation for analysis without generating an animation.
    Returns the final state of agents_list and their trajectories.
    """
    agents_list = []
    for _ in range(num_agents):
        # Random initial position within the arena (not too close to the edge)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, arena_radius - agent_radius - 1)
        start_x = radius * np.cos(angle)
        start_y = radius * np.sin(angle)

        agents_list.append({
            'pos': np.array([start_x, start_y], dtype=float),
            'controller': EnhancedBurstyBRGM(goal_strength=0.03, brm_strength=0.97),
            'trajectory': [np.array([start_x, start_y], dtype=float)]
        })

    for frame in range(num_steps):
        for i, agent in enumerate(agents_list):
            wall_dists_list, goal_dist = sense_agent(agent['pos'])
            agent['controller'].step((wall_dists_list, goal_dist))
            agent['controller'].update_components(agent['pos'], goal_position)

            agent['pos'] += agent['controller'].motor_command * 0.5
            agent['pos'] = arena_boundaries(agent['pos'])
            agent['trajectory'].append(agent['pos'].copy())

    return agents_list

def run_simulation_and_collect_components(num_agents, num_steps, arena_radius, agent_radius, goal_radius, goal_position):
    """
    Runs the multi-agent simulation and collects time-series data for
    primary goal-seeking velocity components and bRGM output contribution components
    for each agent.
    """
    agents_data = []
    for _ in range(num_agents):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, arena_radius - agent_radius - 1)
        start_x = radius * np.cos(angle)
        start_y = radius * np.sin(angle)

        agents_data.append({
            'pos': np.array([start_x, start_y], dtype=float),
            'controller': EnhancedBurstyBRGM(goal_strength=0.03, brm_strength=0.97),
            'primary_vel_x_history': [],
            'primary_vel_y_history': [],
            'brm_output_x_history': [],
            'brm_output_y_history': []
        })

    for frame in range(num_steps):
        for i, agent_data in enumerate(agents_data):
            wall_dists_list, goal_dist = sense_agent(agent_data['pos'])
            agent_data['controller'].step((wall_dists_list, goal_dist))
            agent_data['controller'].update_components(agent_data['pos'], goal_position)

            # Store component data
            agent_data['primary_vel_x_history'].append(agent_data['controller'].primary_velocity[0])
            agent_data['primary_vel_y_history'].append(agent_data['controller'].primary_velocity[1])
            agent_data['brm_output_x_history'].append(agent_data['controller'].brm_output[0])
            agent_data['brm_output_y_history'].append(agent_data['controller'].brm_output[1])

            # Update position
            agent_data['pos'] += agent_data['controller'].motor_command * 0.5
            agent_data['pos'] = arena_boundaries(agent_data['pos'])

    return agents_data
