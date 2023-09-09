"""
Script for downloading, preprocessing the original dataset
and saving it saving it to the path specified in the config file 'prepared_data'
Contains a function get_dataset() that takes as an argument the path to the source dataset for:
loading particular columns of data;
changing types of the column ('descriptions') to string;
removing duplicate title and descriptions columns;
removing short descriptions (150 characters,
to avoid the error that occurs when the original description is less than the set summary max_length);
creating needed an 'id' column for data mapping.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

option_path = './my_module/options/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)

original_data_path = option['original_data_path']
prepared_data_path = option['prepared_data_path']

def get_dataset(data_path):
    dataset = pd.read_csv(data_path, usecols =['title', 'description', 'points','variety','price'], encoding='latin1')
    dataset["description"] = dataset["description"].astype(pd.StringDtype())
    dataset['len'] = dataset['description'].str.len()
    dataset.drop_duplicates(subset=['title','description'], keep=False, inplace = True)
    # To reduce the dataset, I removed descriptions of more than 400 characters, but this is not necessary
    dataset = dataset.loc[(dataset['len'] > 150) & (dataset['len'] < 400)]
    dataset = dataset.dropna().reset_index().reset_index()
    dataset.drop('index', axis=1,inplace = True)
    dataset.dropna().reset_index(inplace = True)
    dataset.rename(columns = {'level_0':'id'}, inplace = True )
    dataset.drop('len', axis=1,inplace = True)
    return dataset

get_dataset(original_data_path).to_csv(prepared_data_path, sep='\t')
