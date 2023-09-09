
"""The data entered by the user enters the program in the form of a string
   The input description is supposed to be in English and not very short
   (at least 100 characters) as it goes through the summary function
"""
import model_builder
import yaml
import torch
import torch.cuda as cuda

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = model_builder.get_tokenizer()
summarizer = model_builder.get_text_summarizer()

option_path = './my_module/options/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)


def get_describe():
    while True:
        try:
            text = str(input("Enter a description of the desired wine: "))
        except ValueError:
            print("Enter description as text")
            continue
        if 0 <= len(text) <= 100:
            print ('The description is very short, please write more information')
        else:
            break
    return text

def get_input_tokens():
    text = get_describe()

    #Get summary of rhe input
    summary = summarizer(text,min_length = option['min_length'],max_length = option['max_length'])[0]["summary_text"]

    # Tokenize the summary
    tokens = tokenizer(summary, truncation=True, padding=True)

    # Convert the token IDs to a tensor
    input_summary_tensor = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(device)

    return (text, summary, input_summary_tensor)
