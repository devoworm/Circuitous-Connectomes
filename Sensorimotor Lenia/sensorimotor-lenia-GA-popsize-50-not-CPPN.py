## A minimal version of Sensorimotor Lenia using a GA in lieu of a CPPN/Deep approach.

import numpy as np
import matplotlib.pyplot as plt

## Simulation parameters
GRID_SIZE = 16
POPULATION_SIZE = 50 # Increased population size for observing genetic diversity
GENOME_SIZE = 5  # Parameters influencing behavior

## Initialize agents
agents = []
for i in range(POPULATION_SIZE):
    x, y = np.random.randint(0, GRID_SIZE, 2)
    genome = np.random.rand(GENOME_SIZE)  # Random genome
    excitation = 0.5 + np.random.rand() * 0.2  # Initial excitation
    agents.append({'pos': np.array([x, y], dtype=int), 'excitation': excitation, 'genome': genome})

## Goal: all agents should have x > GRID_SIZE//2 (right half)
GOAL_X = GRID_SIZE // 2

def sense(agent, agents, grid):
    """Agent senses nearby agents and grid boundaries."""
    x, y = agent['pos']
    neighbors = []
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                # Check for other agents in the neighbor cell
                has_other_agent = False
                for other in agents:
                    if (other['pos'][0], other['pos'][1]) == (nx, ny) and other is not agent:
                        has_other_agent = True
                        break
                if has_other_agent:
                    neighbors.append(1.0)
                else:
                    neighbors.append(0.0)
            else:
                neighbors.append(-1.0)  # Boundary
    return np.array(neighbors)

def act(agent, genome):
    """
    Given the genome, decide how to move.
    Genome[0]: bias right
    Genome[1]: bias up
    Genome[2]: sensitivity to neighbors
    Genome[3]: decay factor
    Genome[4]: exploration noise
    """
    # Simple linear controller
    x, y = agent['pos']
    sensing = sense(agent, agents, None) # Pass agents for sense
    # Combine genome and sensing into a direction vector
    move_x = genome[0] * 0.5 + genome[2] * (np.sum(sensing) - len(sensing)/2) + np.random.randn() * genome[4] * 0.2
    move_y = genome[1] * 0.5 + np.random.randn() * genome[4] * 0.2
    # Normalize so we don't move too far
    norm = np.linalg.norm([move_x, move_y])
    if norm > 0:
        move_x /= norm
        move_y /= norm
    return np.array([move_x, move_y])

def update_agent(agent, genome, agents):
    # Sense environment
    sensing = sense(agent, agents, None)
    # Decide movement
    move = act(agent, genome)
    # Update position (clip to grid)
    new_pos = agent['pos'] + np.round(move).astype(int)
    new_pos = np.clip(new_pos, 0, GRID_SIZE-1)
    agent['pos'] = new_pos.astype(int)
    # Excitation decay and spread
    agent['excitation'] *= genome[3]  # decay
    # Optional: excitation spreads to neighbors (not implemented for brevity)
    # Optional: if agent reaches goal, reward

def evaluate_population(agents):
    # Fraction of agents in right half
    count_right = sum(1 for a in agents if a['pos'][0] > GOAL_X)
    return count_right / len(agents)

def mutate(genome):
    # Simple mutation: add Gaussian noise
    return genome + np.random.randn(GENOME_SIZE) * 0.1

def crossover(parent1, parent2):
    # Blend two genomes
    return 0.5 * parent1 + 0.5 * parent2

# Genetic algorithm loop
NUM_GENERATIONS = 200 # Increased number of generations for long-term convergence
avg_scores_history = [] # To store average scores per generation

for gen in range(NUM_GENERATIONS):
    # Let's associate each agent with its genome
    agent_genomes = [a['genome'].copy() for a in agents]
    agent_scores = []
    for i, agent in enumerate(agents):
        # Run a few steps with this genome
        # Reset agent position for evaluation to avoid cumulative effects across agents in a single generation
        initial_pos = np.random.randint(0, GRID_SIZE, 2)
        agent['pos'] = initial_pos # Reset position for each agent evaluation

        for step_eval in range(10):  # Evaluate over 10 steps
            update_agent(agent, agent_genomes[i], agents)
        # Score: proximity to goal (right side)
        score = 1.0 if agent['pos'][0] > GOAL_X else 0.0 + (agent['pos'][0] - GOAL_X)/GRID_SIZE * 0.5
        agent_scores.append(score)

    # Select top genomes (elitism)
    idx_sorted = np.argsort(agent_scores)[::-1]
    elite_genomes = [agent_genomes[i].copy() for i in idx_sorted[:2]]  # Keep top 2

    # Generate new genomes via crossover and mutation
    new_genomes = []
    for _ in range(POPULATION_SIZE):
        # Ensure p1 and p2 are valid indices for elite_genomes
        p1_idx = np.random.randint(0, len(elite_genomes))
        p2_idx = np.random.randint(0, len(elite_genomes))
        child = crossover(elite_genomes[p1_idx], elite_genomes[p2_idx])
        child = mutate(child)
        new_genomes.append(child)

    # Assign new genomes to agents
    for i, agent in enumerate(agents):
        agent['genome'] = new_genomes[i]
        agent['excitation'] = 0.5  # Reset excitation

    # Print progress and store average score
    avg_score = np.mean(agent_scores)
    avg_scores_history.append(avg_score)
    print(f"Gen {gen+1}: Avg score = {avg_score:.2f}")

## After evolution, show final positions
for i, agent in enumerate(agents):
    print(f"Agent {i}: pos={agent['pos']}, genome={agent['genome']}")

# Plotting the average score history
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_GENERATIONS + 1), avg_scores_history, marker='o')
plt.title('Average Agent Score Over Generations')
plt.xlabel('Generation')
plt.ylabel('Average Score')
plt.grid(True)
plt.show()
