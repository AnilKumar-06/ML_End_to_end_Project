import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("split into train and test.")
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                'Random Forest':RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regressor': LinearRegression(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=False),
                'AdaBoost Regressor':AdaBoostRegressor()
            }
            
            params = {
                
                'Random Forest':{
                    #'criterian':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators':[8, 16, 32, 64, 128]
                },
                
                 'Decision Tree':{
                    #'criterian':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt', 'log2']
                },
                 
                'Gradient Boosting':{
                    'learning_rate':[0.1, 0.01, 0.05],
                    'n_estimators':[8, 16, 32, 64, 128]
                },
                
                'Linear Regressor':{},
                
                'XGBRegressor':{
                    'learning_rate':[0.1, 0.01, 0.05],
                    'n_estimators':[8, 16, 32, 64, 128]
                },
                
                'CatBoostRegressor':{
                    'depth':[6, 8],
                    'learning_rate':[0.1, 0.01, 0.05],
                },
                
                'AdaBoost Regressor':{
                    'learning_rate':[0.1,0.01,0.5],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128]
                }
                
            }
            
            model_report:dict = evaluate_models(x_train, y_train, x_test, y_test,
                                                models, params)
            
            best_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model = models[best_model_name]
            
            if best_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best model Found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj = best_model
            )
            
            pred = best_model.predict(x_test)
            r2Score = r2_score(y_test, pred)
            return r2Score
        
        except Exception as e:
            raise CustomException(e, sys)