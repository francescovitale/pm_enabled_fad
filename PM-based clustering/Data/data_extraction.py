import numpy as np
import pandas as pd
import sys
import os
import random
import matplotlib.pyplot as plt

input_dir = "Input/"
output_dir = "Output/"
output_train_dir = output_dir + "Train/"
output_test_dir = output_dir + "Test/"
output_global_dir = output_dir + "Global/"
output_plots_dir = output_dir + "Plots/"

# Ensure output directories exist
for path in [output_train_dir, output_test_dir, output_global_dir, output_plots_dir]:
    os.makedirs(path, exist_ok=True)

def load_timeseries(n_samples_per_opmode):
    timeseries = {}
    for ds_filename in os.listdir(input_dir):
        if not ds_filename.endswith('.csv'):
            continue
        
        df = pd.read_csv(input_dir + ds_filename)
        if ds_filename.split(".")[0].lower() == "vibration":
            timeseries["OP_MODE_0"] = df.loc[df["label"] == "working"].reset_index(drop=True).drop(columns="label", axis=1)["value"]
            timeseries["OP_MODE_1"] = df.loc[df["label"] == "proximity"].reset_index(drop=True).drop(columns="label", axis=1)["value"]
            timeseries["OP_MODE_2"] = df.loc[df["label"] == "brake"].reset_index(drop=True).drop(columns="label", axis=1)["value"]
        elif ds_filename.split(".")[0].lower() == "x_raw_64":
            valid_rows = df.loc[(df.iloc[:, 0] > -32678) & (df.iloc[:, 0] < 32677)]
            timeseries["OP_MODE_3"] = valid_rows.iloc[:, 0].reset_index(drop=True)
        elif ds_filename.split(".")[0].lower() == "x_raw_heavy_64":
            valid_rows = df.loc[(df.iloc[:, 0] > -32678) & (df.iloc[:, 0] < 32677)]
            timeseries["OP_MODE_4"] = valid_rows.iloc[:, 0].reset_index(drop=True)

    # Apply sample limits if defined
    if n_samples_per_opmode != -1:
        for key in timeseries:
            timeseries[key] = timeseries[key][:n_samples_per_opmode]
            
    return timeseries

def feature_extraction(timeseries):
    extended_timeseries = {}
    for op_mode in timeseries:
        extended_timeseries_window = pd.DataFrame(columns=["max_vibration", "min_vibration", "mean_vibration", "std_vibration"])
        max_values, min_values, mean_values, std_values = feature_extractor(list(timeseries[op_mode]), 10, True)
        extended_timeseries_window["max_vibration"] = max_values[:,0]
        extended_timeseries_window["min_vibration"] = min_values[:,0]
        extended_timeseries_window["mean_vibration"] = mean_values[:,0]
        extended_timeseries_window["std_vibration"] = std_values[:,0]
        extended_timeseries[op_mode] = extended_timeseries_window
    return extended_timeseries

def feature_extractor(data, window_size, overlap):
    max_array, min_array, mean_array, std_array = [], [], [], []
    i = 0
    while i < len(data):
        window = data[i:i+window_size]
        if len(window) == window_size:
            max_array.append(np.max(window))
            min_array.append(np.min(window))
            mean_array.append(np.mean(window))
            std_array.append(np.std(window))
        if overlap:
            i += 1
        else:
            i += window_size

    max_array = np.reshape(np.array(max_array),(len(max_array),1))
    min_array = np.reshape(np.array(min_array),(len(min_array),1))
    mean_array = np.reshape(np.array(mean_array),(len(mean_array),1))
    std_array = np.reshape(np.array(std_array),(len(std_array),1))

    return max_array, min_array, mean_array, std_array

def normalize(timeseries):
    normalized_timeseries = {}
    for op_mode in timeseries:
        df = timeseries[op_mode].copy()
        for column in df.columns:
            col_vals = df[column].values
            mean, std = np.mean(col_vals), np.std(col_vals)
            std = 1 if std == 0 else std
            df[column] = (col_vals - mean) / std
        normalized_timeseries[op_mode] = df
    return normalized_timeseries

def inject_synthetic_anomalies(df, ratio=0.15):
    """Injects SUBTLE synthetic anomalies into a random subset of features."""
    df = df.copy()
    df['is_anomaly'] = 0  # 0 for normal ground truth
    
    n_anomalies = int(len(df) * ratio)
    anomaly_indices = random.sample(list(df.index), n_anomalies)
    
    features = ["max_vibration", "min_vibration", "mean_vibration", "std_vibration"]
    for idx in anomaly_indices:
        # Instead of spiking all features, randomly pick 1 to 3 features to spike
        num_features_to_spike = random.randint(1, 3)
        features_to_spike = random.sample(features, num_features_to_spike)
        
        for feature in features_to_spike:
            # Shift the normal Z-score by a smaller margin (1.0 to 2.5 std devs)
            spike = random.choice([-1, 1]) * random.uniform(1.0, 2.5)
            df.at[idx, feature] += spike
            
        df.at[idx, 'is_anomaly'] = 1  # 1 for anomaly ground truth
        
    return df

def plot_sensor_data(df, title, filename, num_samples=1000, show_anomalies=False):
    """Plots the time-series features. Optionally highlights injected anomalies."""
    features = ["max_vibration", "min_vibration", "mean_vibration", "std_vibration"]
    
    # We plot a subset of data to keep the visualization clean and readable
    plot_df = df.head(num_samples).reset_index(drop=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    for i, feature in enumerate(features):
        axes[i].plot(plot_df.index, plot_df[feature], label="Normal", color="steelblue", alpha=0.7)
        
        # Scatter the anomalies in red if applicable
        if show_anomalies and 'is_anomaly' in plot_df.columns:
            anomaly_indices = plot_df[plot_df['is_anomaly'] == 1].index
            axes[i].scatter(anomaly_indices, plot_df.loc[anomaly_indices, feature], 
                            color="red", label="Anomaly", zorder=5, s=20)
            
        axes[i].set_ylabel(feature)
        axes[i].legend(loc="upper right")
        
    axes[-1].set_xlabel("Time (Window Samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_dir, filename), dpi=150)
    plt.close()

def map_nodes_and_save(timeseries):
    op_modes = list(timeseries.keys())
    if len(op_modes) < 5:
        print("Error: Did not find all 5 OP Modes in the Input folder.")
        sys.exit()

    # Shuffle to randomly assign OP modes to nodes
    random.shuffle(op_modes)
    global_mode = op_modes.pop(0)
    
    # 1. Save and Plot Global Node (100% of its data, unmodified standard scaling)
    timeseries[global_mode].to_csv(output_global_dir + "GLOBAL_NODE.csv", index=False)
    plot_sensor_data(timeseries[global_mode], "Sensor Data - GLOBAL NODE", "GLOBAL_NODE_plot.png", show_anomalies=False)
    
    # 2. Assign remaining 4 modes to Local Nodes and Save/Plot Training Data
    local_data_dict = {}
    for idx, mode in enumerate(op_modes):
        node_name = f"NODE_{idx}"
        
        local_data_dict[node_name] = timeseries[mode].copy()
        
        # Save 100% of this node's data for Training
        local_data_dict[node_name].to_csv(output_train_dir + f"{node_name}.csv", index=False)
        plot_sensor_data(local_data_dict[node_name], f"Training Data - {node_name}", f"{node_name}_train_plot.png", show_anomalies=False)

    # 3. Create, Plot, and Save Cross-Node Test Data 
    for node_name in local_data_dict.keys():
        # Pick a DIFFERENT node to act as the base test data
        other_nodes = [n for n in local_data_dict.keys() if n != node_name]
        target_node = random.choice(other_nodes)
        
        # Take 100% of the other node's data
        test_df = local_data_dict[target_node].copy()
        
        # Inject Subtle Synthetic Anomalies (~15%)
        test_df = inject_synthetic_anomalies(test_df, ratio=0.15)
        
        # Plot the anomalies BEFORE shuffling so the time-series structure is visible
        plot_sensor_data(test_df, f"Test Data (with Anomalies) - {node_name}", f"{node_name}_test_plot.png", show_anomalies=True)
        
        # Shuffle for the actual test set
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        
        # Save as this node's Test set
        test_df.to_csv(output_test_dir + f"{node_name}_test.csv", index=False)

# ==========================================
# Script Execution
# ==========================================
try:
    n_samples_per_opmode = int(sys.argv[1])
except IndexError:
    print("Usage: python data_extraction.py <n_samples_per_opmode>")
    sys.exit()

# Run Pipeline
timeseries = load_timeseries(n_samples_per_opmode)
timeseries = feature_extraction(timeseries)
timeseries = normalize(timeseries)
map_nodes_and_save(timeseries)

print("Data extraction, scaling, testing split, anomaly injection, and distinct plotting completed successfully.")