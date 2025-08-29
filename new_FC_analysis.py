import os
import numpy as np
import pandas as pd
import networkx as nx
import nilearn.image as nimg
import nilearn.input_data as ninput
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BIDSloop:
    def __init__(self, base_dir):
        logging.info("Initializing BIDSloop class.")
        self.base_dir = base_dir
        self.bold_files = self._index_files()

    def _index_files(self):
        logging.info("Indexing BOLD files.")
        return [os.path.join(root, file)
                for root, _, files in os.walk(self.base_dir)
                for file in files if file.endswith("desc-preproc_bold.nii.gz")]

    def get_bold_files(self, subject_id=None):
        logging.info(f"Fetching BOLD files for subject: {subject_id}")
        if subject_id:
            return [f for f in self.bold_files if subject_id in f]
        return self.bold_files

class brain_atlas:
    def __init__(self, atlas_path=None, atlas_name=None):
        logging.info("Initializing brain_atlas class.")
        if atlas_path:
            self.atlas_img = nimg.load_img(atlas_path)
        elif atlas_name:
            self.atlas_img = self.fetch_atlas(atlas_name)
        else:
            raise ValueError("Either an atlas path or an atlas name must be provided.")
    
    def fetch_atlas(self, atlas_name):
        logging.info(f"Fetching atlas: {atlas_name}")
        atlas_fetchers = {
            "craddock": lambda: datasets.fetch_atlas_craddock_2012(
              data_dir=None, 
              url="http://cluster_roi.projects.nitrc.org/Parcellations/craddock_2011_parcellations.tar.gz", 
              resume=True, 
              verbose=1, 
              homogeneity=None, 
              grp_mean=True
            )
        }
        if atlas_name in atlas_fetchers:
            atlas = atlas_fetchers[atlas_name]()
            return nimg.index_img(atlas['tcorr_mean'], 42)
        else:
            raise ValueError(f"Atlas '{atlas_name}' not recognized.")

class process:
    def __init__(self, atlas_img, bold_file):
        logging.info("Initializing process class.")
        self.atlas_img = atlas_img
        self.bold_img = nimg.load_img(bold_file)
    
    def resample_atlas(self):
        logging.info("Resampling atlas to match BOLD image.")
        return nimg.resample_to_img(self.atlas_img, self.bold_img, interpolation='nearest')
    
    def extract_time_series(self):
        logging.info("Extracting time series.")
        resampled_atlas_img = self.resample_atlas()
        masker = ninput.NiftiLabelsMasker(labels_img=resampled_atlas_img, standardize=True, detrend=True, low_pass=0.1, high_pass=0.01, t_r=3.0)
        time_series = masker.fit_transform(self.bold_img)
        logging.info("Time series extraction complete.")
        return time_series
    
    def compute_correlation_matrix(self, time_series):
        logging.info("Computing correlation matrix.")
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        return correlation_matrix
    
    def compute_adjacency_matrix(self, correlation_matrix, threshold=0.2):
        logging.info(f"Computing adjacency matrix with threshold {threshold}.")
        adjacency_matrix = (correlation_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        adjacency_matrix = np.triu(adjacency_matrix)
        return adjacency_matrix
    
    def create_network(self, adjacency_matrix):
        logging.info("Creating network from adjacency matrix.")
        G = nx.from_numpy_array(adjacency_matrix)
        return G
    
    def compute_degree_distribution(self, G):
        logging.info("Computing degree distribution.")
        degrees = [d for _, d in G.degree()]
        hist = np.histogram(degrees, bins=range(0, max(degrees)+2))
        print("\ndd done\n")
        return hist
    
    def compute_modularity(self, G):#, subject_id, output_dir):
        logging.info("Computing modularity.")
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
            print("\nmodularity done\n")
            return modularity
        except ImportError:
            logging.warning("Modularity computation skipped due to missing community_louvain module.")
            return None    

    def compute_graph_metrics(self, G, subject_id, output_dir):
        dd = self.compute_degree_distribution(G)[0]
        metrics = {
            "no_nodes": G.number_of_nodes(),
            "no_edges": G.number_of_edges(),
            "avg_degree": np.mean([d for _, d in G.degree()]),
            "variance_degree_distribution": np.var([d for _, d in G.degree()]),
            "shannon_entropy_degree_distribution": entropy(dd[dd > 0] / np.sum(dd), base=2),
            "avg_clustering_coeff": nx.average_clustering(G),
            "avg_modularity": self.compute_modularity(G),
            "avg_shortest_path": nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
        }
        
        centralities = {
            "degree_distribution": dd,
            "degree_centrality": nx.degree_centrality(G),
            "betweenness_centrality": nx.betweenness_centrality(G),
            "closeness_centrality": nx.closeness_centrality(G),
            "eigenvector_centrality": nx.eigenvector_centrality(G, max_iter=1000),
            "katz_centrality": nx.katz_centrality(G, alpha=1/ max(np.linalg.eigvals(nx.to_numpy_array(G))), max_iter=1000, tol=1e-6)
        }
        

        for key, values in centralities.items():
            # Check if values are a dictionary (e.g., degree_centrality returns a dictionary)
            if isinstance(values, dict):
                values = list(values.values())  # Convert dictionary values to a list

            # If values are a numpy array (e.g., eigenvector centrality), handle directly
            elif isinstance(values, np.ndarray):
                values = values.flatten()  # Flatten in case it's a multidimensional array

            avg_value = np.mean(values)  # Now calculate the mean of the values
            metrics[f"avg_{key}"] = avg_value

            # Create a DataFrame with the node index and corresponding centrality values
            df = pd.DataFrame({
                "Node": range(len(values)),  # Generate node index
                key: values  # Directly use the centrality values
            })

            # Save the DataFrame if needed
            df.to_csv(os.path.join(output_dir, f"sub-{subject_id}_{key}.csv"), index=False)

        return metrics


bids_dir = "/home/udit/Desktop/1TB/derivative_ABIDE-II_BIDS/ABIDEII-NYU_1-BIDS"
output_dir = "./network_metrics"
participants_df = pd.read_csv("/home/udit/Desktop/1TB/data_ABIDE-II_BIDS/ABIDEII-NYU_1-BIDS/participants.tsv", sep="\t")

os.makedirs(output_dir, exist_ok=True)

bids_data = BIDSloop(bids_dir)
bold_files = bids_data.get_bold_files()
atlas = brain_atlas(atlas_name="craddock")

all_metrics = []
for bold_file in bold_files:
    subject_id = os.path.basename(bold_file).split("_")[0]
    subject_id_numeric = subject_id.replace("sub-", "")
    matched_row = participants_df[participants_df["participant_id"] == f"sub-G{subject_id_numeric}"]
    if matched_row.empty:
        logging.warning(f"No matching subject found for {subject_id}")
        continue
    group = matched_row["group"].values[0]
    subject_output_dir = os.path.join(output_dir, group)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    processor = process(atlas.atlas_img, bold_file)
    time_series = processor.extract_time_series()
    correlation_matrix = processor.compute_correlation_matrix(time_series)
    adjacency_matrix = processor.compute_adjacency_matrix(correlation_matrix)
    G = processor.create_network(adjacency_matrix)
    metrics = processor.compute_graph_metrics(G, subject_id, subject_output_dir)
    all_metrics.append(metrics)

pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "network_parameters.csv"), index=False)



'''
# Example usage
# Select the first BOLD file to test
bold_file = bold_files[0]

# Extract the subject_id from the selected BOLD file
subject_id = os.path.basename(bold_file).split("_")[0]
subject_id_numeric = subject_id.replace("sub-", "") 
print(subject_id_numeric)
# Find the matching row for the subject in the participants DataFrame
matched_row = participants_df[participants_df["participant_id"] == f"sub-G{subject_id_numeric}"]
if matched_row.empty:
    logging.warning(f"No matching subject found for {subject_id}")
else:
    group = matched_row["group"].values[0]

    # Process the selected BOLD file
    processor = process(atlas.atlas_img, bold_file)
    time_series = processor.extract_time_series()
    correlation_matrix = processor.compute_correlation_matrix(time_series)
    adjacency_matrix = processor.compute_adjacency_matrix(correlation_matrix)
    G = processor.create_network(adjacency_matrix)
    metrics = processor.compute_graph_metrics(G, subject_id)  # No output directory needed now
'''
