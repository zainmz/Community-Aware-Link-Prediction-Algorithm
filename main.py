import os
import pickle
import networkx as nx
from community import community_louvain
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt

# Load the graph data from the given file
G = nx.read_edgelist('facebook_combined.txt', nodetype=int)

# Pre-compute neighbors for each node
neighbors_dict = {node: set(G.neighbors(node)) for node in G.nodes()}

# Community Detection using the Louvain Method
communities = community_louvain.best_partition(G)

# Feature Extraction Functions for Link Prediction

# Jaccard Coefficient
def get_jaccard(G, u, v):
    preds = nx.jaccard_coefficient(G, [(u, v)])
    return [p for _, _, p in preds][0]

# Adamic-Adar Index
def get_adamic_adar(G, u, v):
    preds = nx.adamic_adar_index(G, [(u, v)])
    return [p for _, _, p in preds][0]

# Check if two nodes belong to the same community
def get_same_community(u, v, communities):
    return 1 if communities[u] == communities[v] else 0

# Compute the strength of the community to which the nodes belong
def get_community_strength(u, v, communities, neighbors_dict):
    u_neighbors = neighbors_dict[u]
    v_neighbors = neighbors_dict[v]
    community_nodes = set([node for node, community in communities.items() if community == communities[u]])
    return len(u_neighbors.intersection(v_neighbors).intersection(community_nodes))

# Check if pre-processed features exist_
if os.path.exists('features.pkl'):
    # Load file
    with open('features.pkl', 'rb') as file:
        data = pickle.load(file)
        X = data['features']
        y = data['labels']
else:
    # Sample a fraction of edges and non-edges for feature extraction
    sample_size = 1
    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))
    sampled_edges = random.sample(edges, int(sample_size * len(edges)))
    sampled_non_edges = random.sample(non_edges, int(sample_size * len(non_edges)))

    # Extract features for each pair of nodes
    X, y = [], []
    for index, (u, v) in enumerate(sampled_edges + sampled_non_edges):
        features = [
            get_jaccard(G, u, v),
            get_adamic_adar(G, u, v),
            get_same_community(u, v, communities),
            get_community_strength(u, v, communities, neighbors_dict)
        ]
        X.append(features)
        y.append(1 if (u, v) in sampled_edges else 0)
        print(f"Processed {index + 1}/{len(sampled_edges) + len(sampled_non_edges)} node pairs.")

    # Save to file
    with open('features.pkl', 'wb') as file:
        pickle.dump({'features': X, 'labels': y}, file)

# Split data into training and testing sets
Z_train, Z_test, a_train, a_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if a pre-trained model exists
if os.path.exists('model.pkl'):
    # Load the trained model
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)
else:
    # Train a new model
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(Z_train, a_train)
    # Save model
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)

# Predict potential links
a_pred = classifier.predict(X)
print(f"Number of potential future links: {sum(a_pred)}")

# Visualization of Predicted links
sample_size = 1
edges = list(G.edges())
non_edges = list(nx.non_edges(G))
sampled_non_edges = random.sample(non_edges, int(sample_size * len(non_edges)))
predicted_links = [pair for i, pair in enumerate(sampled_non_edges) if a_pred[i] == 1]

# Draw the original graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=10, node_color='blue', edge_color='gray')

# Overlay the predicted links on the same graph
nx.draw_networkx_edges(G, pos, edgelist=predicted_links, edge_color='red', width=0.5)

plt.title("Predicted Links Highlighted in Red")
plt.show()
