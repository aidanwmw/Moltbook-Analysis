import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import random
from tqdm import tqdm
from utils import load_moltbook_graph, load_reddit_graph

print("Loading datasets...")

# 1. Load Datasets and Build Graphs via utils.py
G_replies = load_moltbook_graph()
G_reddit = load_reddit_graph()

print(f"Moltbook replies:         {G_replies.number_of_nodes()} nodes, {G_replies.number_of_edges()} edges")
print(f"Reddit replies:           {G_reddit.number_of_nodes()} nodes, {G_reddit.number_of_edges()} edges")


# 3. Extract Features for Random Forest
def extract_ego_features(ego_g):
    """Extract graph-level features from an ego graph for Random Forest"""
    H = ego_g.to_undirected() if ego_g.is_directed() else ego_g
    
    if H.number_of_nodes() < 3:
        return None
        
    # Get center node (ego)
    # The ego node is typically the first node or the one with highest degree in a radius 1/2 graph
    # We approximate by taking the node with max degree
    degrees = dict(H.degree())
    ego_node = max(degrees, key=degrees.get)
    
    features = {
        'num_nodes': H.number_of_nodes(),
        'num_edges': H.number_of_edges(),
        'density': nx.density(H),
        'ego_degree': degrees[ego_node],
        'ego_clustering': nx.clustering(H, ego_node),
        'avg_clustering': nx.average_clustering(H),
        'assortativity': nx.degree_assortativity_coefficient(H) if H.number_of_edges() > 1 else 0,
    }
    
    # Handle NaN values that might arise from empty or peculiar graphs
    for k, v in features.items():
        if np.isnan(v):
            features[k] = 0
            
    return features

def sample_ego_features(G, n_samples=2000, radius=2, label=0, desc=""):
    H = G.to_undirected() if G.is_directed() else G
    nodes = list(H.nodes())
    random.shuffle(nodes)
    
    data = []
    with tqdm(total=n_samples, desc=desc) as pbar:
        for node in nodes:
            ego = nx.ego_graph(H, node, radius=radius)
            features = extract_ego_features(ego)
            
            if features:
                features['label'] = label
                data.append(features)
                pbar.update(1)
                
            if len(data) >= n_samples:
                break
                
    return pd.DataFrame(data)

# Extract features
print("Sampling ego graphs and extracting features...")
moltbook_df = sample_ego_features(G_replies, n_samples=2000, radius=2, label=0, desc="Moltbook")
reddit_df = sample_ego_features(G_reddit, n_samples=2000, radius=2, label=1, desc="Reddit")

# Combine datasets
df_all = pd.concat([moltbook_df, reddit_df], ignore_index=True)

# Define X and y
X = df_all.drop('label', axis=1)
y = df_all['label']

# 4. Train and Tune Random Forest
print(f"Features loaded. Shape: {X.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Starting Hyperparameter Tuning...")
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# 5. Evaluate
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1] # Probability estimates for the positive class
acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
class_report = classification_report(y_test, y_pred, target_names=["Moltbook (0)", "Reddit (1)"])
print(class_report)

print("Exporting cross-validation logs and best parameters...")
# Save GridSearch results and summary log
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv("outputs/rf_tuning_results.csv", index=False)

with open("rf_analysis_log.txt", "w") as f:
    f.write("=== Random Forest Baseline Analysis Log ===\n\n")
    f.write(f"Total ego graphs extracted: {len(df_all)}\n")
    f.write("Parameters Scanned:\n")
    for k, v in param_grid.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nBest Parameters Found: {grid_search.best_params_}\n")
    f.write(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}\n")
    f.write(f"\nTest Set Accuracy: {acc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(class_report)

# 6. Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Feature Importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

ax1.set_title("Feature Importances")
ax1.bar(range(X.shape[1]), importances[indices], align="center")
ax1.set_xticks(range(X.shape[1]))
ax1.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
ax1.set_xlim([-1, X.shape[1]])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Moltbook", "Reddit"])
disp.plot(ax=ax2, cmap=plt.cm.Blues)
ax2.set_title("Confusion Matrix")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Receiver Operating Characteristic')
ax3.legend(loc="lower right")

plt.tight_layout()
plt.savefig("outputs/rf_baseline_results.png", dpi=300)
print("Saved performance plot to rf_baseline_results.png")
