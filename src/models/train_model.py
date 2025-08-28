import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from numpy.typing import NDArray
import pickle
import logging
import os


def load_dataset(path:str)-> tuple[NDArray[np.float64],NDArray[np.float64]]:
    try:
        train_tfidf=pd.read_csv(path)
        # print(train_tfidf.shape)
        x_train=train_tfidf.drop("target",axis=1).values
        y_train=train_tfidf["target"].values
        return x_train,y_train
    except FileNotFoundError as e:
        print(e)
        raise
def train_model(x_train:NDArray,y_train:NDArray)->MultinomialNB:
    mnb=MultinomialNB()
    mnb.fit(x_train,y_train)
    return mnb

def save_model(path:str,model_obj:MultinomialNB)->None:
    try:
     os.makedirs(os.path.dirname(path),exist_ok=True)
     with open(path,"wb") as f:
        pickle.dump(model_obj,f)
    except FileNotFoundError as e:
       print(e)
       raise

def main():
   x_train,y_train=load_dataset("data/processed/train_tfidf.csv")
   mnb=train_model(x_train,y_train)
   save_model("models/model.pkl",mnb)

if __name__=="__main__" :
   main()