import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
import pm4py.algo.conformance.tokenreplay as tokenreplay

import pm4py
import sys
import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

# ==========================================
# 1. Configuration & Paths
# ==========================================
input_dir = "Input/"
input_globalnode_dir = input_dir + "Global/"  
input_localnodes_dir = input_dir + "Train/"   

output_dir = "Output/"
output_petrinets_dir = output_dir + "PetriNets/"
output_petrinets_localnodes_dir = output_petrinets_dir + "LocalNodes/"
output_petrinets_globalnode_dir = output_petrinets_dir + "GlobalNode/"
output_similaritymatrix_dir = output_dir + "SimilarityMatrix/"

# New paths for Event Logs
output_eventlogs_dir = output_dir + "EventLogs/"
output_eventlogs_localnodes_dir = output_eventlogs_dir + "LocalNodes/"
output_eventlogs_globalnode_dir = output_eventlogs_dir + "GlobalNode/"

# Ensure output directories exist so the script doesn't crash on save
for path in [
    output_petrinets_localnodes_dir, 
    output_petrinets_globalnode_dir, 
    output_similaritymatrix_dir,
    output_eventlogs_localnodes_dir,
    output_eventlogs_globalnode_dir
]:
    os.makedirs(path, exist_ok=True)

# ==========================================
# 2. Helper Functions
# ==========================================
def read_node_data():
    global_node = None
    local_nodes = {}
    
    for node in os.listdir(input_globalnode_dir):
        if node.endswith(".csv"):
            global_node = pd.read_csv(input_globalnode_dir + node)
            
    for node in os.listdir(input_localnodes_dir):
        if node.endswith(".csv"):
            local_nodes[node.split(".csv")[0]] = pd.read_csv(input_localnodes_dir + node)

    return global_node, local_nodes

def cluster_node_data(global_node, local_nodes, n_clusters, clustering_technique):
    clustered_global_node = None
    clustered_local_nodes = {}

    n_features = len(global_node.columns)
    starting_centroids = [[0.0] * n_features for _ in range(n_clusters)]

    # 1. Cluster the GLOBAL node FIRST and capture the parameters
    clustered_global_node, global_clustering_params = cluster_dataset(
        global_node, 
        reuse_parameters=0, 
        starting_centroids=starting_centroids, 
        n_clusters=n_clusters, 
        clustering_technique=clustering_technique, 
        clustering_parameters_in=None
    )   

    # 2. Force all LOCAL nodes to use the exact same global parameters
    for node in local_nodes:
        clustered_local_nodes[node], _ = cluster_dataset(
            local_nodes[node], 
            reuse_parameters=1, 
            starting_centroids=None, 
            n_clusters=n_clusters, 
            clustering_technique=clustering_technique, 
            clustering_parameters_in=global_clustering_params
        )   

    return clustered_global_node, clustered_local_nodes

def cluster_dataset(dataset, reuse_parameters, starting_centroids, n_clusters, clustering_technique, clustering_parameters_in):
    clustered_dataset = dataset.copy()
    clustering_parameters = {}

    if reuse_parameters == 0:
        if clustering_technique == "gmm":
            gaussian_mixture = GaussianMixture(n_components=n_clusters, means_init=starting_centroids, covariance_type='full', random_state=0).fit(dataset)
            clustering_parameters["weights"] = gaussian_mixture.weights_
            clustering_parameters["means"] = gaussian_mixture.means_
            clustering_parameters["covariances"] = gaussian_mixture.covariances_
            labels_probability = gaussian_mixture.predict_proba(dataset)
            for j in range(n_clusters):
                temp = []
                for i in range(0,len(dataset)):
                    temp.append(labels_probability[i][j])
                clustered_dataset["Cluster_" + str(j)] = temp
            cluster_ids = []
            for row_idx,row in clustered_dataset.iterrows():
                clusters = []
                for i in range(0, n_clusters):
                    clusters.append(row["Cluster_" + str(i)])
                cluster_ids.append(clusters.index(max(clusters)))
            for i in range(0,n_clusters):
                clustered_dataset = clustered_dataset.drop(axis=1, columns="Cluster_"+str(i))
            clustered_dataset["Cluster"] = cluster_ids

        elif clustering_technique == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters, init=np.array(starting_centroids), random_state=0).fit(clustered_dataset)
            cluster_labels = kmeans.labels_
            clustered_dataset["Cluster"] = cluster_labels
            cluster_labels = cluster_labels.tolist()
            used = set()
            clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

            instances_sets = {}
            centroids = {}
            
            for cluster in clusters:
                instances_sets[cluster] = []
                centroids[cluster] = []
                
            temp = clustered_dataset
            for index, row in temp.iterrows():
                instances_sets[int(row["Cluster"])].append(row.values.tolist())
            
            n_features_per_instance = len(instances_sets[0][0])-1
            
            for instances_set_label in instances_sets:
                instances = instances_sets[instances_set_label]
                for idx, instance in enumerate(instances):
                    instances[idx] = instance[0:n_features_per_instance]
                for i in range(0,n_features_per_instance):
                    values = []
                    for instance in instances:
                        values.append(instance[i])
                    centroids[instances_set_label].append(np.mean(values))
                    
            clustering_parameters = centroids
            
    else:
        if clustering_technique == "gmm":
            gaussian_mixture = GaussianMixture(n_components=n_clusters, covariance_type='full')
            gaussian_mixture.weights_ = clustering_parameters_in["weights"]
            gaussian_mixture.means_ = clustering_parameters_in["means"]
            gaussian_mixture.covariances_ = clustering_parameters_in["covariances"]
            gaussian_mixture.precisions_cholesky_ = _compute_precision_cholesky(clustering_parameters_in["covariances"], 'full')
            labels_probability = gaussian_mixture.predict_proba(dataset)
            for j in range(n_clusters):
                temp = []
                for i in range(0,len(dataset)):
                    temp.append(labels_probability[i][j])
                clustered_dataset["Cluster_" + str(j)] = temp
            cluster_ids = []
            for row_idx,row in clustered_dataset.iterrows():
                clusters = []
                for i in range(0, n_clusters):
                    clusters.append(row["Cluster_" + str(i)])
                cluster_ids.append(clusters.index(max(clusters)))
            for i in range(0,n_clusters):
                clustered_dataset = clustered_dataset.drop(axis=1, columns="Cluster_"+str(i))
            clustered_dataset["Cluster"] = cluster_ids

        elif clustering_technique == "kmeans":
            clusters = []
            for index, instance in clustered_dataset.iterrows():
                min_value = float('inf')
                min_centroid = -1
                for centroid in clustering_parameters_in:
                    centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
                    dist = np.linalg.norm(instance.values-centroid_coordinates)
                    if dist<min_value:
                        min_value = dist
                        min_centroid = centroid
                clusters.append(min_centroid)
            
            clustered_dataset["Cluster"] = clusters
        
    return clustered_dataset, clustering_parameters

def extract_node_event_logs(clustered_global_node, clustered_local_nodes):
    global_node_event_log = None
    local_node_event_logs = {}
    
    clustered_node_state_transitions = extract_state_transitions(clustered_global_node)
    global_node_event_log = build_event_log(clustered_node_state_transitions)
        
    for node in clustered_local_nodes:
        clustered_node_state_transitions = extract_state_transitions(clustered_local_nodes[node])
        local_node_event_logs[node] = build_event_log(clustered_node_state_transitions)
    
    return global_node_event_log, local_node_event_logs

def extract_state_transitions(data):
    state_transitions = []
    current_state = None
    
    for idx,sample in data.iterrows():
        if current_state == None:
            current_state = sample["Cluster"]
        else:
            next_state = sample["Cluster"]
            if next_state != current_state:
                state_transitions.append(str(int(current_state)) + "_to_" + str(int(next_state)))
                current_state = next_state
    
    return state_transitions

def build_event_log(state_transitions):
    event_log = []
    for idx,state_transition in enumerate(state_transitions):
        caseid = 0
        event_timestamp = timestamp_builder(idx)
        event = [caseid, state_transition, event_timestamp]
        event_log.append(event)

    event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
    event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
    event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
    event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
    event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        
    return event_log    

def timestamp_builder(number):
    ss = number
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)
    ignore, hh = divmod(hh, 24)
    
    ss = ss%60
    mm = mm%60
    hh = hh%24
    
    return "1900-01-01T"+str(hh).zfill(2)+":"+str(mm).zfill(2)+":"+str(ss).zfill(2)

def build_node_petri_nets(process_discovery_variant, local_node_event_logs, global_node_event_log):
    global_petri_net = None
    local_petri_nets = {}
    
    global_petri_net = process_discovery(global_node_event_log, process_discovery_variant)
    
    for node_event_log in local_node_event_logs:
        local_petri_nets[node_event_log] = process_discovery(local_node_event_logs[node_event_log], process_discovery_variant)
    
    return local_petri_nets, global_petri_net

def process_discovery(event_log, variant):
    petri_net = {}

    if variant == "im":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log)
    elif variant == "imf25":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.25)
    elif variant == "imf50":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.50)
    elif variant == "imf75":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.75)
    elif variant == "imf99":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.99)
    elif variant == "ilp":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.00)
    elif variant == "ilp25":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.25)
    elif variant == "ilp50":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.50)
    elif variant == "ilp75":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.75)
    elif variant == "ilp99":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.99)
    elif variant == "alpha":
        petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_alpha(event_log)

    return petri_net

def extract_similarity_vecs(local_node_event_logs, global_petri_net):
    node_similarity_vecs = {}
    nodes_fitness = {}

    for node_event_log in local_node_event_logs:
        nodes_fitness[node_event_log], _ = compute_fitness(global_petri_net, local_node_event_logs[node_event_log], "ALIGNMENT_BASED")
        print(nodes_fitness[node_event_log])
    
    for reference_node in nodes_fitness:
        node_similarity_vecs[reference_node] = []
        reference_fitness_value = nodes_fitness[reference_node]
        other_fitness_values = {}
        
        for other_node in nodes_fitness:
            if other_node != reference_node:
                other_fitness_values[other_node] = nodes_fitness[other_node]       
                
        for other_node in other_fitness_values:
            other_fitness_values[other_node] = 1 - abs(reference_fitness_value - other_fitness_values[other_node])
            
        reference_fitness_value = 1.0
        normalizing_value = sum(list(other_fitness_values.values()) + [reference_fitness_value])
        
        node_similarity_vecs[reference_node].append([reference_node, reference_fitness_value / normalizing_value])
        for other_node in other_fitness_values:
            node_similarity_vecs[reference_node].append([other_node, other_fitness_values[other_node] / normalizing_value])

    return node_similarity_vecs
    
def compute_fitness(petri_net, event_log, cc_variant):
    log_fitness = 0.0
    aligned_traces = None
    parameters = {}
    parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
    
    if cc_variant == "ALIGNMENT_BASED":
        aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
        log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
    elif cc_variant == "TOKEN_BASED":
        replay_results = tokenreplay.algorithm.apply(log = event_log, net = petri_net["network"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
        log_fitness = replay_fitness.evaluate(results = replay_results, variant = replay_fitness.Variants.TOKEN_BASED)["log_fitness"]

    return log_fitness, aligned_traces  

def save_event_logs(global_node_event_log, local_node_event_logs):
    """Exports the generated PM4Py event logs to standard .xes files."""
    xes_exporter.apply(global_node_event_log, output_eventlogs_globalnode_dir + "GLOBAL_NODE.xes")
    
    for node, log in local_node_event_logs.items():
        xes_exporter.apply(log, output_eventlogs_localnodes_dir + node + ".xes")

def save_similarity_matrix(node_similarity_vecs):
    node_ids = list(node_similarity_vecs.keys())
    for idx, node_id in enumerate(node_ids):
        node_ids[idx] = int(node_id.split("_")[-1])
        
    node_ids.sort()
    for idx, node_id in enumerate(node_ids):
        node_ids[idx] = "NODE_" + str(node_ids[idx])
        
    similarity_matrix = pd.DataFrame(columns=node_ids, index=node_ids)
    for node in node_similarity_vecs:
        for pair in node_similarity_vecs[node]:
            similarity_matrix[node][pair[0]] = pair[1]
        
    similarity_matrix.to_csv(output_similaritymatrix_dir + "SimilarityMatrix.csv")
    return None

def save_petri_nets(local_petri_nets, global_petri_net):
    for node in local_petri_nets:
        pnml_exporter.apply(local_petri_nets[node]["network"], local_petri_nets[node]["initial_marking"], output_petrinets_localnodes_dir + node + ".pnml", final_marking = local_petri_nets[node]["final_marking"])
    
    pnml_exporter.apply(global_petri_net["network"], global_petri_net["initial_marking"], output_petrinets_globalnode_dir + "GLOBAL_NODE.pnml", final_marking = global_petri_net["final_marking"])
    return None

# ==========================================
# 3. Execution Execution
# ==========================================
try:
    n_clusters = int(sys.argv[1])
    clustering_technique = sys.argv[2]
    process_discovery_variant = sys.argv[3]
except IndexError:
    print("Enter the right number of input arguments: <n_clusters> <clustering_technique> <process_discovery_variant>")
    sys.exit()

global_node, local_nodes = read_node_data()

# Ensure we actually loaded data before proceeding
if global_node is None or not local_nodes:
    print("Error: Could not load data. Check if Input/Global/GLOBAL_NODE.csv and Input/Train/NODE_X.csv exist.")
    sys.exit()

clustered_global_node, clustered_local_nodes = cluster_node_data(global_node, local_nodes, n_clusters, clustering_technique)

# Generate Event Logs
global_node_event_log, local_node_event_logs = extract_node_event_logs(clustered_global_node, clustered_local_nodes)

# Save the Event Logs
save_event_logs(global_node_event_log, local_node_event_logs)

# Build and Save Petri Nets
local_petri_nets, global_petri_net = build_node_petri_nets(process_discovery_variant, local_node_event_logs, global_node_event_log)
save_petri_nets(local_petri_nets, global_petri_net)

# Compute and Save Similarity Matrix
node_similarity_vecs = extract_similarity_vecs(local_node_event_logs, global_petri_net)
save_similarity_matrix(node_similarity_vecs)

print(f"Event Logs, Petri Nets, and Similarity Matrix saved successfully to {output_dir}")