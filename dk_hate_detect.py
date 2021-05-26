#! usr/bin/python

#import general data handling libraries
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

#import BERT related libraries and functions
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
import transformers as tf
import torch

#Class for converting tokenized text into a tensor which can be used for prediction
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings #Define encodings
        self.labels = labels #define labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} #Define tensor item
        if self.labels: #if labels are included, add to tensor
            item["labels"] = torch.tensor(self.labels[idx]) 
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    

#Class for detecting hatespeech in either a single string or in a column
class detect_hate:
    
    def __init__(self, args):
        self.args = args #Add args to class
        #Load tokenizer from huggingface
        self.tokenizer = AutoTokenizer.from_pretrained("Guscode/DKbert-hatespeech-detection") 
        #Load model from huggingface
        self.model = tf.AutoModelForSequenceClassification.from_pretrained("Guscode/DKbert-hatespeech-detection")
        #Define Trainer from model
        self.text_trainer=Trainer(self.model)
    
    
    def single_string_classification(self): #Function for performing single string classification
        single_text = list([self.args["text"]]) #Read string as list
        single_text_tokenized = self.tokenizer(single_text, padding=True, truncation=True, max_length=128) #tokenize

       
        text_dataset = Dataset(single_text_tokenized)  # Create torch dataset
        raw_pred, _, _ = self.text_trainer.predict(text_dataset) #Predict whether string is hateful or not
        
        if np.argmax(raw_pred[0]) == 0: #if string is not hateful:
            hate_or_not = "not hateful"
        else: #If string is hateful:
            hate_or_not = "hateful"
        
        print(f"this text is {hate_or_not}")
        
    def add_hate_column(self): #Function for performing hatespeech classification on a column in a dataset
        data = pd.read_csv(self.args["data"]) #Read data
        texts = list(data[self.args["column"]]) #select the right column
        texts_tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=128) #tokenize the texts

        text_dataset = Dataset(texts_tokenized) # Create torch dataset
        raw_pred, _, _ = self.text_trainer.predict(text_dataset) #Predict whether each string is hateful or not
        
        hate_column = "_".join([self.args["column"], "hateful"]) #create name of hate column
        
        data[hate_column] = [np.argmax(pred) for pred in raw_pred] #Add column to dataset
        
        #create output path
        output_path = os.path.join(self.args["output"], "_".join([Path(self.args["data"]).stem, "withhate.csv"]))

        data.to_csv(output_path) #Save dataset to .csv
        
        print(f"data saved to: {output_path}")
       
    
def main():
    #Add all the terminal arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required = False, default = None,
                    help="Path to the a dataset in csv format")
    
    ap.add_argument("--column", required = False, default = None,
                    help="name of column including text for hatespeech detection")
    
    ap.add_argument("--text", required = False, default = None, type = str,
                    help="string for single string hatespeech detection")
    
    ap.add_argument("--output",required = False, type = str, default = "./",
                    help="output path for dataset with hatespeech column")
    #parse arguments
    args = vars(ap.parse_args())
    
    
    hater = detect_hate(args) #Initialize class
    if args["text"] is not None: #Perform single string classification
        hater.single_string_classification()
    elif args["data"] is not None: #perform classification on column in dataset
        hater.add_hate_column()
    else:
        print("add some text please")

if __name__=="__main__":
    main()
        
        