import pandas as pd

## 1. Degree Centrality
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)

## Convert to DataFrame for easy sorting and display
in_degree_df = pd.DataFrame(in_degree_centrality.items(), columns=['Node', 'In-Degree Centrality']).sort_values(by='In-Degree Centrality', ascending=False)
out_degree_df = pd.DataFrame(out_degree_centrality.items(), columns=['Node', 'Out-Degree Centrality']).sort_values(by='Out-Degree Centrality', ascending=False)

print("\n--- Top 5 Nodes by In-Degree Centrality ---")
display(in_degree_df.head())

print("\n--- Top 5 Nodes by Out-Degree Centrality ---")
display(out_degree_df.head())

## 2. Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
betweenness_df = pd.DataFrame(betweenness_centrality.items(), columns=['Node', 'Betweenness Centrality']).sort_values(by='Betweenness Centrality', ascending=False)

print("\n--- Top 5 Nodes by Betweenness Centrality ---")
display(betweenness_df.head())

## 3. Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)
closeness_df = pd.DataFrame(closeness_centrality.items(), columns=['Node', 'Closeness Centrality']).sort_values(by='Closeness Centrality', ascending=False)

print("\n--- Top 5 Nodes by Closeness Centrality ---")
display(closeness_df.head())

import random
import math

## 4. Defining a Spatiotemporal Agent
class TemporalSpatialAgent:
    def __init__(self, name="Agent_01"):
        self.name = name
        # PAST: Memory buffer representing historical trajectory and internal models [3, 11]
        self.past_memory = []
        # PRESENT: Immediate state (x, y) and current sensorimotor bias [3, 12]
        self.present_pos = (0, 0)
        self.heading = 0 # In degrees

    def perceive_present(self):
        """Simulates immediate sensory input (Present)."""
        # Adding 'circuitous' noise to model the non-optimal sensory signal [6, 7]
        drift = random.uniform(-5, 5)
        self.heading += drift
        print(f"[Present] Heading: {self.heading:.2f} at {self.present_pos}")

    def generate_alternate_futures(self, branching_factor=3):
        """Projects multiple potential trajectories (Alternate Futures) [4]."""
        alternate_futures = []
        for i in range(branching_factor):
            # Each path is a 'Rube Goldberg' step into a potential outcome [13, 14]
            future_drift = random.uniform(-30, 30)
            future_heading = self.heading + future_drift
            # Calculate a potential next step
            future_x = self.present_pos[0] + math.cos(math.radians(future_heading))
            future_y = self.present_pos[1] + math.sin(math.radians(future_heading))
            alternate_futures.append((future_x, future_y))

        print(f"[Future] Branching into {branching_factor} alternate futures.")
        return alternate_futures

    def update_past(self):
        """Stores the current state into the Past for recursive feedback [3]."""
        self.past_memory.append(self.present_pos)
        # Check for 'Loop Closure': Recognition of a previously visited landmark [9, 15]
        for idx, prev_pos in enumerate(self.past_memory[:-1]):
            dist = math.dist(self.present_pos, prev_pos)
            if dist < 0.5:
                print(f"[Past] Loop Closure Detected at step {idx}! Updating internal map.")

    def navigate_circuitously(self):
        """The navigation loop incorporating maximum intermediate steps [5]."""
        # Step 1: Process Past (Memory/Referents) [16]
        self.update_past()

        # Step 2: Sense Present (Symbol/Input) [16]
        self.perceive_present()

        # Step 3: Evaluate Futures (Understanding/Prediction) [16]
        futures = self.generate_alternate_futures()

        # Select one future to become the new present (Decision via internal model) [17]
        self.present_pos = random.choice(futures)

## Execution of Step 4.
nav_model = TemporalSpatialAgent()
for step in range(5):
    print(f"--- Navigation Step {step} ---")
    nav_model.navigate_circuitously()
