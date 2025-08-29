import pickle 
import json
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.base import BaseEstimator
from typing import Dict
import logging
import mlflow
import mlflow.sklearn

# seted tracking uri 

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# started logging 

logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

# console_handler

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file_handler

file_handler=logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_test_dataset(path:str)->pd.DataFrame:
    try:
       test_data=pd.read_csv(path)
       logger.debug("test data retreived from %s",path)

       x_test=test_data.drop(columns=["target"],axis=1)
       y_test=test_data["target"]
       return x_test,y_test
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def load_model(path:str)->BaseEstimator:
    try:
        with open(path,"rb") as f:
           model=pickle.load(f)
           logger.debug("model loaded from %s",path)
        return model
    except FileNotFoundError as e:
        logger.error("file not found %s",e)
    except Exception as e:
        logger.error("any unexpected error accured %s",e)
        raise

def prediction(model:BaseEstimator,x_test:np.ndarray)->np.ndarray:
    
    try:
        logger.info("predictio started")
        y_pred=model.predict(x_test)
        return y_pred
    except Exception as e:
        logger.error("somthing error accured %s",e)
        raise
logger.info("predicted sucessfully")

def evaluation(y_pred:np.ndarray,y_test:np.ndarray)->Dict[str,float]:
    try:
       logger.info("model evaluation start...")
       accuracy=accuracy_score(y_test,y_pred)
       precision=precision_score(y_test,y_pred)
       recall=recall_score(y_pred,y_test)
       f1_scr=f1_score(y_pred,y_test)
       metrics={
          "accuracy":accuracy,
          "precision":precision,
          "recall": recall,
          "f1_scr": f1_scr
     }
       return metrics
    except Exception as e:
        logger.error("something error accured %s",e)
        raise
logger.info("returned matrices sucessfully")


def save_metrics(metrics:Dict[str,float],path:str)->None:
    try:
        logger.info("directory making...")
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"w") as f:
            json.dump(metrics,f,indent=4)
        logger.debug("eval mertric saved at %s",path) 
    except Exception as e:
        logger.error("somthing error accured %s",e)  
        raise 

def save_model_info(run_id:str,model_path:str,file_path:str):
    try:
        model_info={"run_id":run_id,"model_path":model_path}
        with open(file_path,"w") as file:
            json.dump(model_info,file,indent=4)
            logger.debug("file saved at %s",file_path)
    except FileNotFoundError as e:
        logger.error("file note found %s",e)  
    except Exception as e:
        logger.error("something unexpected error accured %s",e)  



def main():
    
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:

       try:
            x_testing,y_testing=load_test_dataset("data/processed/test_tfidf.csv")
            x_test=x_testing.values
            y_test=y_testing.values


            model=load_model("models/model.pkl")

            y_pred=prediction(model,x_test)

            metrics = evaluation(y_pred,y_test)
       
           # log metrics at mlflow 

            save_metrics(metrics,"reports/metrics.json")
            for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
            
           # Log model parameters to MLflow
            if hasattr(model, 'get_params'):
               params = model.get_params()
               for param_name, param_value in params.items():
                  mlflow.log_param(param_name, param_value)
                
            input_example = pd.DataFrame(x_test, columns=x_testing.columns).iloc[:5]
            signature = mlflow.models.signature.infer_signature(input_example, model.predict(x_test[:5]))
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
            

            save_model_info(run.info.run_id,"model","reports/exp_info.json")

            mlflow.log_artifact("reports/metrics.json")

            mlflow.log_artifact("reports/exp_info.json")

            mlflow.log_artifact("model_evaluation_errors.log")

       except Exception as e:
           logger.error("something error accured %s",e)
           raise
if __name__=="__main__":
    main()


