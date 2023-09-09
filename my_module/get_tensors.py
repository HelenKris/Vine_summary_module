"""
Contains a function 'get_tensors' that takes as input the path to the dataset,
the batch size and the path where the resulting tensors will be saved
and gets a dictionary with tensors and their ID numbers (as keys)
obtained from all summary descriptions of wine.
The function also saves a dictionary of the resulting tensors to the path 'tensors_path' from the config file.
The load_tensors function takes the path with the stored dictionary as an argument and loads
the tensors into the loaded_summary_tensors variable on available devise and returns it.
"""

from pathlib import Path
import model_builder
import data_setup
import torch
import torch.cuda as cuda
import yaml
import pandas as pd

option_path = './my_module/options/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)

device = "cuda" if torch.cuda.is_available() else "cpu"

summarizer = model_builder.get_summarizer()

tensors_path = option['tensors_path']
prepared_data_path = option['prepared_data_path']
batch_size = option['batch_size']

def get_tensors(prepared_data_path,batch_size,tensors_path):
    dataset = pd.read_csv(prepared_data_path, sep='\t',dtype={"summary":"string"})
    num_rows = len(dataset)
    batch_size = batch_size
    summary_tensors = {}
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = dataset[start:end]
        batch_summary_tensors = {}

        for index, row in batch.iterrows():
            index_id = row['id']
            text = row['description']
            summary = summarizer(text)
            summary_tensor = summary[0]['summary_token_ids'].unsqueeze(0).to(device)
            batch_summary_tensors[index_id] = summary_tensor

        summary_tensors.update(batch_summary_tensors)

    # Save summary tensors to a pkl file
    with open(tensors_path, "wb") as f:
        torch.save(summary_tensors, f)

def load_tensors(tensors_path):
    with open(tensors_path, "rb") as f:
        # loaded_summary_tensors
        if cuda.is_available():
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cuda'))
        else:
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cpu'))

    return loaded_summary_tensors


def load_tensors():
    with open(tensors_path, "rb") as f:
        # loaded_summary_tensors
        if cuda.is_available():  # Check if CUDA is available before loading the tensors
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cuda'))
        else:
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cpu'))

    return loaded_summary_tensors
