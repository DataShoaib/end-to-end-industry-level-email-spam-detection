import pandas as pd
import numpy as np
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

# -----------start-------- #

logger=logging.getLogger("data_ingesion")
logger.setLevel(logging.DEBUG)

# making formatter 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
# add handler
logger.addHandler(console_handler)

def load_yaml():
    with open("params.yaml","r") as f:
        params=yaml.safe_load(f)
        return params


# loading dataset


def data_load(file_path:str)-> pd.DataFrame:
    try:
        df=pd.read_csv(file_path,encoding="latin-1")
        # df.dropna(axis=0,how='any',inplace=True)
        return df
    except FileNotFoundError as e:
        logger.error("filenotfound %s",e)
        raise
    except Exception as e:
        logger.error("any unexepted error accured during data loading %s",e)
        raise
logger.debug("dataframe retreived")


# slightly cleaning


def removed_irrelavent_clm(df:pd.DataFrame)->pd.DataFrame:
    try:
       df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
       df.duplicated().sum()
       df.rename(columns={"v1":"target","v2":"text"},inplace=True)
       return df
    except KeyError as e:
        logger.error("unknow column %s",e)
        raise
    except Exception as e:
        logger.error("any unexpected error %s",e)
        raise
logger.debug("silghtly cleaning completed")   


# saving final dataset


def save_df(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
       raw_data_path=os.path.join(data_path,"raw")
       os.makedirs(raw_data_path,exist_ok=True)    
       train_data.to_csv(os.path.join(raw_data_path,"train_data.csv"),index=False)
       test_data.to_csv(os.path.join(raw_data_path,"test_data.csv"),index=False)
    except Exception as e:
        logger.error("any unexpected error accurred %s",e)
        raise  


def main():
    params=load_yaml()
    df=data_load("data/raw/spam.csv")
    df=removed_irrelavent_clm(df)
    train_data,test_data=train_test_split(df,test_size=params["data_ingetion"]["test_size"],random_state=params["data_ingetion"]["random_state"])
    save_df(train_data,test_data,"data")
logger.debug("df save completed")    


if __name__=="__main__":
    main()