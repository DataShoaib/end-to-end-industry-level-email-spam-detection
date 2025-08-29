import mlflow
import logging
import json

mlflow.set_tracking_uri('http://127.0.0.1:5000')

logger=logging.getLogger("model_registery")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_model_info(file_path:str)->json:
    try:
       with open(file_path,"r") as file:
           model_info=json.load(file)
           logger.debug("file loaded sucessfully from %s",file_path)
           return model_info
    except FileNotFoundError as e:
        logger.error("file not found at %s",e)
    except Exception as e:
        logger.error("something unexpected error accured %s",e)
        raise

def register_model(model_info:json,model_name=str):
    try:
       #  regiter model
       register_model=mlflow.register_model(model_uri=model_info,name=model_name,tags={"author":"shoaib"})

       #  transition to staging

       client=mlflow.tracking.MlflowClient()
       client.transition_model_version_stage(
          name=model_name,
          version=register_model.version,
          stage="Staging"
        )
       logger.debug(f"model {model_name} version {register_model.version} registred sucessgully and transition to staging")
    except Exception as e:
        logger.error("something error accured during model registration %s",e)


def main():
    model_info=load_model_info("reports/exp_info.json")
    model_name="email_spam_det_MulNB"
    if model_info:
       register_model(model_info,model_name)

if __name__=="__main__":
    main()