import nltk
import os
import string
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def load_data(train_data_path:str,test_data_path:str)->pd.DataFrame:
    proce_train_data=pd.read_csv(train_data_path)
    proce_test_data=pd.read_csv(test_data_path)
    return proce_train_data,proce_test_data

def clean_text(text:str)->None:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    #  Join back to string
    return " ".join(stemmed)




def save_df(train_processed:pd.DataFrame,test_processed:pd.DataFrame,saving_path:str)->None:
    raw_data_path=os.path.join(saving_path,"interim")
    os.makedirs(raw_data_path,exist_ok=True)
    train_processed.to_csv(os.path.join(raw_data_path,"train_processed.csv"),index=False)
    test_processed.to_csv(os.path.join(raw_data_path,"test_processed.csv"),index=False)

    



def main():
    train_data,test_data=load_data("data/raw/train_data.csv","data/raw/test_data.csv")
    train_processed=train_data.copy()
    test_processed=test_data.copy()
    train_processed["text"]=train_processed["text"].apply(lambda x: clean_text(x))
    test_processed["text"]=test_processed["text"].apply(lambda x: clean_text(x))
    save_df(train_processed,test_processed,"data")   

if __name__=="__main__":
    main()