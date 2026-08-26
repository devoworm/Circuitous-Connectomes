import pandas as pd

# 1. Degree Centrality
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)

# Convert to DataFrame for easy sorting and display
in_degree_df = pd.DataFrame(in_degree_centrality.items(), columns=['Node', 'In-Degree Centrality']).sort_values(by='In-Degree Centrality', ascending=False)
out_degree_df = pd.DataFrame(out_degree_centrality.items(), columns=['Node', 'Out-Degree Centrality']).sort_values(by='Out-Degree Centrality', ascending=False)

print("\n--- Top 5 Nodes by In-Degree Centrality ---")
display(in_degree_df.head())

print("\n--- Top 5 Nodes by Out-Degree Centrality ---")
display(out_degree_df.head())

# 2. Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
betweenness_df = pd.DataFrame(betweenness_centrality.items(), columns=['Node', 'Betweenness Centrality']).sort_values(by='Betweenness Centrality', ascending=False)

print("\n--- Top 5 Nodes by Betweenness Centrality ---")
display(betweenness_df.head())

# 3. Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)
closeness_df = pd.DataFrame(closeness_centrality.items(), columns=['Node', 'Closeness Centrality']).sort_values(by='Closeness Centrality', ascending=False)

print("\n--- Top 5 Nodes by Closeness Centrality ---")
display(closeness_df.head())
