import mlflow 
import logging 


logger=logging.getLogger("promote_model")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
