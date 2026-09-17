import pandas as pd
import numpy as np
import os

def extract_upper_triangle(file_path):
    """
    Extracts the upper triangle of a correlation matrix from a TSV file.
    """
    matrix = pd.read_csv(file_path, sep='\t', header=None).values
    # Get the upper triangle excluding the diagonal
    upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return upper_triangle

def load_data(data_dir):
    """
    Loads all TSV files in a directory and extracts upper triangles.
    Returns a list of vectors and a list of participant IDs.
    """
    vectors = []
    ids = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".tsv"):
            file_path = os.path.join(data_dir, filename)
            vector = extract_upper_triangle(file_path)
            vectors.append(vector)
            
            # Assuming filename is something like 'sub-001.tsv'
            participant_id = filename.split('.')[0]
            ids.append(participant_id)
            
    return np.array(vectors), ids
