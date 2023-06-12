import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig
# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngetionConfig:
    train_path: str = os.path.join('artifacts', 'train.csv')
    test_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'student_data.csv')
    
class DataIngetion:
    def __init__(self):
        self.ingetion_config = DataIngetionConfig()
 
    def initiate_data_ingetion(self):
        logging.info("Entered the data ingetion component!")
        try:
            df = pd.read_csv("artifacts\student_data.csv")
            logging.info("Read the dataset as  dataframe.")
            os.makedirs(os.path.dirname(self.ingetion_config.train_path), exist_ok=True)
            df.to_csv(self.ingetion_config.train_path, index=False, header=True)
            
            logging.info("Train Test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingetion_config.train_path, index=False, header=True)
            test_set.to_csv(self.ingetion_config.test_path, index=False, header=True)
            logging.info('Ingetion of data is completed!')
            
            return(
                self.ingetion_config.train_path,
                self.ingetion_config.test_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        

        