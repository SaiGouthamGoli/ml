import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

        
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Split Train and Test Data")
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])
            
            models = {
                "Linear Regression" : LinearRegression(),
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
            }
            
            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model = models[
                list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                ]
            
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            # Logging
            logging.info("Found Best Model on Train and Test Data")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )
            
            return best_model_score
        # Raise Custom Exception 
        except Exception as e:
            raise CustomException(e, sys)