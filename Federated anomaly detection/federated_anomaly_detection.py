import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve

# ==========================================
# 1. Configuration & Paths
# ==========================================
INPUT_DIR = "Input/"
DATA_DIR = os.path.join(INPUT_DIR, "Data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
GLOBAL_DIR = os.path.join(DATA_DIR, "Global")
SIMILARITY_MATRIX_FILE = os.path.join(INPUT_DIR, "SimilarityMatrix.csv")
OUTPUT_DIR = "Output/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS_PER_ROUND = 25
FEDERATED_ROUNDS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ==========================================
# 2. Autoencoder & Helper Functions
# ==========================================
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim=4):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 2), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_train_data(filepath):
    df = pd.read_csv(filepath)
    features = df[['max_vibration', 'min_vibration', 'mean_vibration', 'std_vibration']].values
    return torch.tensor(features, dtype=torch.float32)

def load_test_data(filepath):
    """Loads features and ground truth labels (is_anomaly) from the test CSV."""
    df = pd.read_csv(filepath)
    features = df[['max_vibration', 'min_vibration', 'mean_vibration', 'std_vibration']].values
    labels = df['is_anomaly'].values
    return torch.tensor(features, dtype=torch.float32), labels

def train_local_model(model, data_tensor, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(epochs):
        for batch_features, _ in loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def evaluate_models(models_dict, test_data_dict):
    """
    Evaluates models on test sets and calculates ROC-AUC, F1-Score, and confusion matrix percentages.
    """
    criterion = nn.MSELoss(reduction='none')
    all_y_true = []
    all_y_scores = []

    with torch.no_grad():
        for node_name, (test_features, test_labels) in test_data_dict.items():
            
            # Select the correct model
            if len(models_dict) == 1:
                model = list(models_dict.values())[0]
            else:
                model = models_dict[node_name]
                
            model.eval()
            reconstructed = model(test_features)
            
            # MSE reconstruction error is the anomaly score
            mse_scores = torch.mean(criterion(reconstructed, test_features), dim=1).numpy()
            
            all_y_true.extend(test_labels)
            all_y_scores.extend(mse_scores)
            
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(all_y_true, all_y_scores)

    # Find the optimal threshold for the F1 score using Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(all_y_true, all_y_scores)
    
    # Calculate F1 scores for all thresholds (adding epsilon to avoid division by zero)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    best_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    # Generate hard binary predictions using the optimal threshold
    y_pred = (all_y_scores >= best_threshold).astype(int)
    
    # Calculate Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(all_y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    
    tp_pct = (tp / total) * 100
    fp_pct = (fp / total) * 100
    tn_pct = (tn / total) * 100
    fn_pct = (fn / total) * 100
    best_f1 = f1_score(all_y_true, y_pred)
    
    return {
        "ROC-AUC": roc_auc,
        "F1-Score": best_f1,
        "TP(%)": tp_pct,
        "FP(%)": fp_pct,
        "TN(%)": tn_pct,
        "FN(%)": fn_pct
    }

# ==========================================
# 3. Main Evaluation Pipeline
# ==========================================
def main():
    print("Loading Data...")
    
    # Load Similarity Matrix
    sim_df = pd.read_csv(SIMILARITY_MATRIX_FILE, index_col=0)
    node_names = sim_df.columns.tolist()

    # Load Train Data
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
    local_train_dict = {f"NODE_{i}": load_train_data(f) for i, f in enumerate(train_files) if f"NODE_{i}" in node_names}
    
    # Load Test Data (Features + Labels)
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.csv")))
    local_test_dict = {f"NODE_{i}": load_test_data(f) for i, f in enumerate(test_files) if f"NODE_{i}" in node_names}
    
    results = {}
    total_epochs = FEDERATED_ROUNDS * EPOCHS_PER_ROUND
    num_nodes = len(local_train_dict)

    # ----------------------------------------------------
    # Strategy 1: No FL (Isolated / Local-Only)
    # ----------------------------------------------------
    print("\n[1/4] Training No FL (Isolated)...")
    isolated_models = {node: AnomalyAutoencoder() for node in local_train_dict}
    for node, data in local_train_dict.items():
        train_local_model(isolated_models[node], data, epochs=total_epochs)
    
    results["No FL (Isolated)"] = evaluate_models(isolated_models, local_test_dict)

    # ----------------------------------------------------
    # Strategy 2: No FL (Centralized Pool)
    # ----------------------------------------------------
    print("[2/4] Training No FL (Centralized)...")
    centralized_model = AnomalyAutoencoder()
    
    # Pool all local training data and the global node data
    all_train_data = list(local_train_dict.values())
    global_file = os.path.join(GLOBAL_DIR, "GLOBAL_NODE.csv")
    if os.path.exists(global_file):
        all_train_data.append(load_train_data(global_file))
        
    centralized_data_tensor = torch.cat(all_train_data, dim=0)
    train_local_model(centralized_model, centralized_data_tensor, epochs=total_epochs)
    
    results["No FL (Centralized)"] = evaluate_models({"Central": centralized_model}, local_test_dict)

    # ----------------------------------------------------
    # Strategy 3: Homogeneous FL (Standard FedAvg)
    # ----------------------------------------------------
    print("[3/4] Training Homogeneous FL (FedAvg)...")
    fedavg_model = AnomalyAutoencoder()
    fedavg_weights = fedavg_model.state_dict()
    
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        local_weights_list = []
        for node, data in local_train_dict.items():
            model = AnomalyAutoencoder()
            model.load_state_dict(fedavg_weights)
            updated_weights = train_local_model(model, data, EPOCHS_PER_ROUND)
            local_weights_list.append(updated_weights)
            
        # Homogeneous Aggregation (1/N weight for all)
        new_fedavg_weights = {}
        for key in fedavg_weights.keys():
            weighted_sum = torch.zeros_like(fedavg_weights[key], dtype=torch.float32)
            for weights in local_weights_list:
                weighted_sum += weights[key] * (1.0 / num_nodes)
            new_fedavg_weights[key] = weighted_sum
            
        fedavg_weights = new_fedavg_weights
        
    fedavg_model.load_state_dict(fedavg_weights)
    results["Homogeneous FL (FedAvg)"] = evaluate_models({"Global": fedavg_model}, local_test_dict)

    # ----------------------------------------------------
    # Strategy 4: Similarity-Weighted FL (Personalized)
    # ----------------------------------------------------
    print("[4/4] Training Similarity-Weighted FL...")
    personalized_models = {node: AnomalyAutoencoder() for node in local_train_dict.keys()}
    personalized_weights = {node: model.state_dict() for node, model in personalized_models.items()}
    
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        trained_local_weights = {}
        
        # Train
        for node_name, data in local_train_dict.items():
            model = personalized_models[node_name]
            model.load_state_dict(personalized_weights[node_name])
            trained_local_weights[node_name] = train_local_model(model, data, EPOCHS_PER_ROUND)

        # Aggregate using SimilarityMatrix.csv
        new_personalized_weights = {}
        for target_node in trained_local_weights.keys():
            new_node_weights = {}
            
            # 1. Fetch the row and normalize it so the weights sum to 1.0
            raw_row = sim_df.loc[target_node]
            normalized_row = raw_row / raw_row.sum() 

            for layer_key in trained_local_weights[target_node].keys():
                weighted_sum = torch.zeros_like(trained_local_weights[target_node][layer_key], dtype=torch.float32)
                
                for source_node in trained_local_weights.keys():
                    # 2. Use the normalized weight instead of the raw similarity score
                    sim_weight = normalized_row[source_node] 
                    weighted_sum += trained_local_weights[source_node][layer_key] * float(sim_weight)
                    
                new_node_weights[layer_key] = weighted_sum
            new_personalized_weights[target_node] = new_node_weights
            
        personalized_weights = new_personalized_weights

    # Load final weights back into models
    for node in personalized_models:
        personalized_models[node].load_state_dict(personalized_weights[node])
        
    results["Similarity-Weighted FL"] = evaluate_models(personalized_models, local_test_dict)

    # ----------------------------------------------------
    # Final Output Report & Saving Results
    # ----------------------------------------------------
    print("\n=================================================================================================")
    print(f"{'Strategy':<30} | {'ROC-AUC':<10} | {'F1-Score':<10} | {'TP (%)':<8} | {'FP (%)':<8} | {'TN (%)':<8} | {'FN (%)':<8}")
    print("=================================================================================================")
    for strategy, metrics in results.items():
        print(f"{strategy:<30} | {metrics['ROC-AUC']:<10.4f} | {metrics['F1-Score']:<10.4f} | "
              f"{metrics['TP(%)']:<8.2f} | {metrics['FP(%)']:<8.2f} | {metrics['TN(%)']:<8.2f} | {metrics['FN(%)']:<8.2f}")
    print("=================================================================================================")

    # Convert results dictionary to Pandas DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Strategy'
    
    # Save to CSV
    output_csv_path = os.path.join(OUTPUT_DIR, "detection_results.csv")
    results_df.to_csv(output_csv_path)
    print(f"\n[INFO] Detection results successfully saved to: {output_csv_path}")

if __name__ == "__main__":
    main()