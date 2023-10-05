import os
import pickle
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the graph data from the given file_
print("Loading graph...")
G = nx.read_edgelist('facebook_combined.txt', nodetype=int)

# Sample a fraction of edges and non-edges for testing
print("Sampling edges and non-edges...")
sample_size = 1
edges = list(G.edges())
non_edges = list(nx.non_edges(G))
sampled_edges = random.sample(edges, int(sample_size * len(edges)))
sampled_non_edges = random.sample(non_edges, int(sample_size * len(non_edges)))

# Check if pre-processed features exist, if yes, load them
if os.path.exists('features.pkl'):
    print("Loading features from file...")
    with open('features.pkl', 'rb') as file:
        data = pickle.load(file)
        X = data['features']
        y = data['labels']

# Split the data into training and testing sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if a pre-trained model exists, if yes, load it
if os.path.exists('model.pkl'):
    print("Loading trained model from file...")
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)

# Use the classifier to predict the links
print("Predicting with CALP algorithm...")
y_pred_calp = classifier.predict(X_test)

# Prepare data for other algorithms
print("Extract data for other algorithms to run...")
df_X = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df_X_test = pd.DataFrame(X_test, columns=['feature1', 'feature2', 'feature3', 'feature4'])
common_rows = df_X.merge(df_X_test, how='inner', left_index=True, right_index=True)
X_test_indices = common_rows.index.tolist()
all_pairs = edges + non_edges
X_test_pairs = [all_pairs[i] for i in X_test_indices]

# Helper function to predict based on a threshold
def predict_with_threshold(scores, threshold=0.5):
    return [1 if score > threshold else 0 for score in scores]

# Helper function to compute common neighbors score
def common_neighbors_score(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))

# Predict using other link prediction algorithms
print("Predicting with Common Neighbors...")
y_pred_common_neighbors = predict_with_threshold([common_neighbors_score(G, u, v) for u, v in X_test_pairs])
print("Predicting with Jaccard Coefficient...")
y_pred_jaccard = predict_with_threshold([p for u, v, p in nx.jaccard_coefficient(G, ebunch=X_test_pairs)])
print("Predicting with Adamic-Adar Index...")
y_pred_adamic_adar = predict_with_threshold([p for u, v, p in nx.adamic_adar_index(G, ebunch=X_test_pairs)])
print("Predicting with Preferential Attachment...")
y_pred_preferential_attachment = predict_with_threshold([p for u, v, p in nx.preferential_attachment(G, ebunch=X_test_pairs)])

# Evaluate and print the performance of each algorithm
print("Evaluating all predictions...")
algorithms = ['CALP', 'Common Neighbors', 'Jaccard', 'Adamic-Adar', 'Preferential Attachment']
predictions = [y_pred_calp, y_pred_common_neighbors, y_pred_jaccard, y_pred_adamic_adar, y_pred_preferential_attachment]

for algo, y_pred in zip(algorithms, predictions):
    print(f"Evaluation for {algo}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print("-" * 50)

print("Done!")
