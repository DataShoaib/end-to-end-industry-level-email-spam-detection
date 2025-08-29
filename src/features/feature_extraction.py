from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import yaml

def load_yamlfile():
    with open("params.yaml","r") as file:
        params=yaml.safe_load(file)
        return params

def load_data(pro_train_data:pd.DataFrame,pro_test_data:pd.DataFrame)->None:
   pro_train_data= pd.read_csv(pro_train_data)
   pro_test_data= pd.read_csv(pro_test_data)
   pro_train_data["text"] = pro_train_data["text"].fillna("")
   pro_test_data["text"] = pro_test_data["text"].fillna("")
   return pro_train_data,pro_test_data

def featue_extraction(pro_train_data:pd.DataFrame,pro_test_data:pd.DataFrame,params:int)->pd.DataFrame:
    max_feat = params["feature_ext"]["max_feature"]
    pro_test_data_targ = pro_test_data["target"]
    pro_train_data_targ= pro_train_data["target"]
    pro_train_data = pro_train_data["text"]
    pro_test_data = pro_test_data["text"]
    vec= TfidfVectorizer(max_features=max_feat)
    train_tf=vec.fit_transform(pro_train_data).toarray()
    test_tf=vec.transform(pro_test_data).toarray()
    train_tfidf=pd.DataFrame(train_tf,columns=vec.get_feature_names_out())
    train_tfidf=pd.concat([train_tfidf,pro_train_data_targ],axis=1)
    train_tfidf["target"]=train_tfidf["target"].map({"ham": 0,"spam": 1})
    test_tfidf=pd.DataFrame(test_tf,columns=vec.get_feature_names_out())
    test_tfidf=pd.concat([test_tfidf,pro_test_data_targ],axis=1)
    test_tfidf["target"]=test_tfidf["target"].map({"ham": 0,"spam": 1})
    return train_tfidf,test_tfidf


def save_df(train_tfidf:pd.DataFrame,test_tfidf:pd.DataFrame,saving_path:str)->None:
    raw_data_path=os.path.join(saving_path,"processed")
    os.makedirs(raw_data_path,exist_ok=True)
    train_tfidf.to_csv(os.path.join(raw_data_path,"train_tfidf.csv"),index=False)
    test_tfidf.to_csv(os.path.join(raw_data_path,"test_tfidf.csv"),index=False)

    
def main():
    params = load_yamlfile()
    pro_train_data,pro_test_data=load_data("data/interim/train_processed.csv","data/interim/test_processed.csv")
    train_tfidf,test_tfidf=featue_extraction(pro_train_data,pro_test_data,params)
    save_df(train_tfidf,test_tfidf,"data")   

if __name__=="__main__":
    main()