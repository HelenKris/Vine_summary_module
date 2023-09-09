"""
Contains functionality for obtaining a dictionary of tensors and summaries of descriptions simultaneously in one pass.
The function is saved as an archive for better times, since it calculates the result very slowly.
"""

from pathlib import Path
import model_builder
import data_setup
import torch
import torch.cuda as cuda
import yaml

option_path = './my_module/options/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = model_builder.get_tokenizer()
summarizer = model_builder.get_text_summarizer()

tensors_path = option['tensors_path']
summaries_path= option['summaries_path']

def get_tensors():
    # Store the summary tensors
    summary_tensors = {}
    # Iterate over the dataset and retrieve summaries
    dataset = data_setup.get_dataset()

    for index, row in dataset.iterrows():
        index_id = row['id']
        text = row['description']

        # Use the summarization pipeline to generate summaries
        summary = summarizer(text,min_length = option['min_length'],max_length = option['max_length'])[0]["summary_text"]

        dataset.at[index, 'summary'] = summary
        # Tokenize the summary
        tokens = tokenizer(summary, truncation=True, padding=True)

        # Convert the token IDs to a tensor
        summary_tensor = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(device)

        # Store the summary tensor with the offer ID as the key
        summary_tensors[index_id] = summary_tensor

        dataset.to_csv(summaries_path, sep='\t')

    # Save the summary tensors to a file
    with open(tensors_path, "wb") as f:
        # pickle.dump(summary_tensors, f)
        torch.save(summary_tensors, f)

    # Load the summary tensors from the file
    with open(tensors_path, "rb") as f:
        # loaded_summary_tensors
        if cuda.is_available():  # Check if CUDA is available before loading the tensors
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cuda'))
        else:
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cpu'))

    # return loaded_summary_tensors
    return len(loaded_summary_tensors) # УДАЛИТЬ ЭТО ДЛЯ ТЕСТИРОВАНИЯ


def load_tensors():
    with open(tensors_path, "rb") as f:
        # loaded_summary_tensors
        if cuda.is_available():  # Check if CUDA is available before loading the tensors
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cuda'))
        else:
            loaded_summary_tensors = torch.load(f, map_location=torch.device('cpu'))

    return loaded_summary_tensors
