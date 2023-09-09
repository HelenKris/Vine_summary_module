# Search_by_summary

This is a claim for an application to search for the type of wine you need by summary of description.
For getting summaries and tensors of token indices here I used summarizing pipeline with model ’bart-large-cnn’.
Then using the cosine distance between the tensors I build a recommendation system for choosing the right wine according to the description entered.

The whole idea is calculating the similarity between descriptions of wine by calculating the cosine distance between tensors obtained from the predictions of a sentence summarization model.
While the summarization pipeline from the Hugging Face transformers library may return not only summary texts but a tensor of token indices as well, I can use these token indices in order to calculating the similarity between descriptions.

So I followed these steps:

1. Iterate through my dataset using the summarization pipeline and get all the tensors of token indices in form of dictionary with keys as IDs. So I can use the offer ID to map these tensors with other information in prepared dataset.

2. Save the summary_tensors to a file: To enable quick loading and comparison, I saved the summary_tensors to a file in a serialized format. I used torch.save().


*In near future I will make awesome readme and docs...but still...*


### Implemented Tasks
1. Text summarizing
2. Recommendation system

## How to start?

1. Config your yaml config file in options
2. python get_me_wine.py
3. Enter a description of the desired wine in English and not very short
   (at least 100 characters) and you will get:
   1. The summary of your description.
   2. The five most relevant wines by summary description with their ID, price and points

## Project Structure

```
Summary_module/
├── experiments/
│   └── Summary_experiments.ipynb
├── my_module/  --main code
│   ├── data_setup.py  --loading datasets and data proccessing
│   ├── model_builder.py  --loads the summarization model with the parameters set in the config file
│   ├── input_setup.py  --input data proccessing
│   ├── get_tensors.py  --getting tensors obtained from the predictions of a sentence sum-n model
│   ├── get_me_wine.py  --main file
│   └── utils.py
├── options/
│   ├── config.yml  --YAML confings for flexible, powerful and easy configuration
├── tensors/
│   └── summary_tensors.pkl
└──  data/
    ├── prepared_data/prepared_data.csv -- prepared dataset ufter data proccessing
    └── train/wine_reviews.csv -- original dataset
