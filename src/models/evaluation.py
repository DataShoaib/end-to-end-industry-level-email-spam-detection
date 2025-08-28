import pickle 
import json
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.base import BaseEstimator
from typing import Dict



def load_test_dataset(path:str)->tuple[np.ndarray,np.ndarray]:
    test_data=pd.read_csv(path)
    x_test=test_data.drop(columns=["target"],axis=1).values
    y_test=test_data["target"].values
    return x_test,y_test

def load_model(path:str)->BaseEstimator:
    with open(path,"rb") as f:
        model=pickle.load(f)
    return model

def prediction(model:BaseEstimator,x_test:np.ndarray)->np.ndarray:
    y_pred=model.predict(x_test)
    return y_pred

def evaluation(y_pred:np.ndarray,y_test:np.ndarray)->tuple[float,float]:
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    # recall=recall_score(y_pred,y_test)
    # f1_scr=f1_score(y_pred,y_test)
    # roc_auc_scr=roc_auc_score(y_pred,y_test)
    return accuracy,precision
def save_metrics(metrics:Dict[str,float],path:str)->None:
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w") as f:
        json.dump(metrics,f,indent=4)

def main():
    x_test,y_test=load_test_dataset("data/processed/test_tfidf.csv")
    model=load_model("models/model.pkl")
    y_pred=prediction(model,x_test)
    accuracy,precision = evaluation(y_pred,y_test)
    save_metrics({"accuracy":accuracy,"precision":precision},"models/metrics.json")

if __name__=="__main__":
    main()


