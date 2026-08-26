!pip install pyvis

import networkx as nx
from pyvis.network import Network
import random

def create_temporal_connectome():
    ## Initialize a Directed Graph
    G = nx.DiGraph()

    ## Define temporal regions (total < 100 neurons)
    past_nodes = [f"Past_{i}" for i in range(15)]
    present_nodes = [f"Present_{i}" for i in range(20)]
    future_nodes = [f"Future_{i}" for i in range(15)]
    intermediate_nodes = [f"Circuitous_{i}" for i in range(30)] # Suboptimal redundancy

    ## Add nodes with temporal metadata
    for node in past_nodes: G.add_node(node, group='Past', color='#FF9999', size=20)
    for node in present_nodes: G.add_node(node, group='Present', color='#99FF99', size=25)
    for node in future_nodes: G.add_node(node, group='Future', color='#9999FF', size=20)
    for node in intermediate_nodes: G.add_node(node, group='Intermediate', color='#CCCCCC', size=15)

    ## 1. Past to Present (Direct and Circuitous)
    for p in past_nodes:
        # Some direct feedforward
        G.add_edge(p, random.choice(present_nodes))
        # Some Rube Goldberg-style circuitous steps (Past -> Intermediate -> Present)
        inter = random.choice(intermediate_nodes)
        G.add_edge(p, inter)
        G.add_edge(inter, random.choice(present_nodes))

    ## 2. Present to Future (Branching Alternate Futures)
    for pr in present_nodes:
        G.add_edge(pr, random.choice(future_nodes))
        # Adding "spaghettification": convolved paths to future outcomes
        inter1 = random.choice(intermediate_nodes)
        inter2 = random.choice(intermediate_nodes)
        G.add_edge(pr, inter1)
        G.add_edge(inter1, inter2)
        G.add_edge(inter2, random.choice(future_nodes))

    ## 3. Feedback Loops (Circular temporal relations: Future -> Past)
    # This represents the "strange loop" where future predictions update past referents
    for f in future_nodes[:5]:
        G.add_edge(f, random.choice(past_nodes))

    # 4. Redundant "Suboptimal" Connections (Bricolage)
    for _ in range(40):
        u, v = random.sample(list(G.nodes), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    ## Create Interactive Visualization
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)
    # Customize physics settings for a more spread out and stable layout
    net.toggle_physics(True)
    net.set_options("""
    var options = {
        "physics": {
            "repulsion": {
                "centralGravity": 0.2,
                "springLength": 200,
                "springConstant": 0.05,
                "nodeDistance": 150,
                "damping": 0.09
            },
            "minVelocity": 0.75,
            "solver": "repulsion",
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "updateInterval": 25
            }
        }
    }
    """)
    net.show("temporal_connectome.html", notebook=False)
    return G

if __name__ == "__main__":
    G = create_temporal_connectome()
    print("Interactive connectome generated as 'temporal_connectome.html'.")

## Calculate graph density
density = nx.density(G)
print(f"Graph Density: {density:.4f}")

## Calculate average clustering coefficient
clustering_coefficient = nx.average_clustering(G)
print(f"Average Clustering Coefficient: {clustering_coefficient:.4f}")
