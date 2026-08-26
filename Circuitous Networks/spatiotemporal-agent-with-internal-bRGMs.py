import numpy as np
import random
import math

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

# Execution
nav_model = TemporalSpatialAgent()
for step in range(5):
    print(f"--- Navigation Step {step} ---")
    nav_model.navigate_circuitously()

class SophisticatedTemporalAgent(TemporalSpatialAgent):
    def __init__(self, state_dim=16):
        super().__init__()
        # RNN-inspired hidden state vector representing internal memory
        self.hidden_state = np.zeros(state_dim)
        # Random weights to simulate a fixed recursive projection
        self.W_h = np.random.randn(state_dim, state_dim) * 0.1
        self.W_x = np.random.randn(state_dim, 2) * 0.1
        # Feedback weights for non-linear recurrence
        self.W_f = np.random.randn(state_dim, state_dim) * 0.05

    def update_past(self):
        """Encoding memory via a non-linear hidden state update with feedback."""
        # Save the raw coordinate (Literal Past)
        super().update_past()

        # Update the Hidden State (Sophisticated Memory)
        input_vec = np.array(self.present_pos)

        # Standard RNN update
        h_prime = np.tanh(np.dot(self.W_h, self.hidden_state) + np.dot(self.W_x, input_vec))

        # Apply Non-linear feedback loop (Self-modulation)
        # h_t = tanh(h' + feedback(h'))
        feedback = np.sin(np.dot(self.W_f, h_prime))
        self.hidden_state = np.tanh(h_prime + feedback)

        print(f"[Memory] Feedback-modulated Hidden State updated. Mean activation: {np.mean(self.hidden_state):.4f}")

# Quick test
soph_nav = SophisticatedTemporalAgent()
for _ in range(3):
    soph_nav.navigate_circuitously()
