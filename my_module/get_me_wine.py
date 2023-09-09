import torch.cuda as cuda
import torch
import yaml
import input_setup
from torch.nn.functional import cosine_similarity
import pandas as pd

original_text, summary, input_summary_tensor = input_setup.get_input_tokens()

option_path = './my_module/options/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)

# Specify the complete file path to load the pkl file
tensors_path = option['tensors_path']
summaries_path= option['summaries_path']
prepared_data_path = option['prepared_data_path']

# dataset = pd.read_csv(summaries_path, sep='\t')
dataset = pd.read_csv(prepared_data_path, sep='\t',dtype={"summary":"string"})

# Check if CUDA is available
device = torch.device("cuda" if cuda.is_available() else "cpu")

# Load the summary tensors from the file
with open(tensors_path, "rb") as f:
    if cuda.is_available():  # Check if CUDA is available before loading the tensors
        loaded_summary_tensors = torch.load(f, map_location=torch.device('cuda'))
    else:
        loaded_summary_tensors = torch.load(f, map_location=torch.device('cpu'))

similarity_scores = {}

for offer_id, summary_tensor in loaded_summary_tensors.items():
        # Adjust the size of tensors to match along the second dimension
    if input_summary_tensor.shape[1] > summary_tensor.shape[1]:
        summary_tensor = torch.nn.functional.pad(summary_tensor, (0, input_summary_tensor.shape[1] - summary_tensor.shape[1]))
    elif input_summary_tensor.shape[1] < summary_tensor.shape[1]:
        input_summary_tensor = torch.nn.functional.pad(input_summary_tensor, (0, summary_tensor.shape[1] - input_summary_tensor.shape[1]))
    # similarity = cosine_similarity(input_summary_tensor.unsqueeze(0).float(), summary_tensor.unsqueeze(0))
    similarity = cosine_similarity(input_summary_tensor.float(), summary_tensor)
    similarity_scores[offer_id] = similarity.item()

# Sort the similarity scores and get the top 5 most similar sentences
top_5_similar_sentences = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]

print(f'Summary: {summary}')
# Print the top 5 most similar sentences along with their similarity scores
for offer_id, similarity in top_5_similar_sentences:
    similarity = round(similarity, 4)
    sentence_text = dataset["description"][offer_id]
    points = dataset["points"][offer_id]
    price = dataset["price"][offer_id]
    print(f"ID: {offer_id} | Points: {points} | Price: {price} |Similarity: {similarity} | Sentence: {sentence_text}")
