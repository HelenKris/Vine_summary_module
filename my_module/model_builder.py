"""
Loads the summarization model with the parameters set in the configuration file
Load the tokenizer for the specific model
The models that this module use is 'bart-large-cnn', you can change this in config.yaml file
See the up-to-date list of available models on huggingface.co/models.
"""
from transformers import AutoTokenizer
from transformers import pipeline
import yaml

option_path = './my_module/options/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)


def get_tokenizer():
    """
    Getting tokenizer for the specific model
    """
    tokenizer = AutoTokenizer.from_pretrained(option['model'])
    return tokenizer

def get_summarizer():
    """
    Loads the summarization model with the parameters set in the configuration file
    This summarization model returns the summary of the corresponding input.
    """
    summarizer = pipeline("summarization", model=option['model'], min_length = option['min_length'],max_length = option['max_length'],return_tensors=True)
    return summarizer

def get_text_summarizer():
    """
    Loads the summarization model with the parameters set in the configuration file.
    This summarization model returns the token ids of the summary as tensor.
    """
    summarizer = pipeline("summarization", model=option['model'], min_length = option['min_length'],max_length = option['max_length'],return_text=True)
    return summarizer
