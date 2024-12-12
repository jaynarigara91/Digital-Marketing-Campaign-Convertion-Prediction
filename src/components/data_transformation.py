import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation paths.
    """
    preprocessor_path_config : str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.prerocessor_path = DataTransformationConfig()

    def get_initialize_data_transformer_obj(self):
        """
        Creates a preprocessing object using pipelines for numeric and categorical columns.
        """
        try:
            # Define numeric and categorical columns
            numeric_columns = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 
                               'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 
                               'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints']
            categorical_columns = ['Gender', 'CampaignChannel', 'CampaignType']

            # Define numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Define categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numeric_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def get_initialize_data_transformation(self,data_path):
        """
        Transforms training and testing data with augmentation (SMOTE).
        """
        try:
            
            data_df = pd.read_csv(data_path)

            logging.info("Load ddata from artifacts folder")

            preprocessing_obj = self.get_initialize_data_transformer_obj()
            
            logging.info("Preprocessing data with pipeline.")

            target_column_name = "Conversion"

            X = data_df.drop(columns=[target_column_name],axis=1)
            y = data_df[target_column_name]
    
            X_transformed = preprocessing_obj.fit_transform(X)
            
            smote = SMOTE(sampling_strategy={0: 4000, 1: 4000}, random_state=42)
            X, y = smote.fit_resample(X_transformed, y)
            
            logging.info("Augmenting data with 8000 Raws.")

            data_arr = np.c_[X,y]

            os.makedirs(os.path.dirname(self.prerocessor_path.preprocessor_path_config), exist_ok=True)
            
            logging.info("Store preprocess.pkl in artifacts folder")
            
            save_object(
                file_path=self.prerocessor_path.preprocessor_path_config,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved at {self.prerocessor_path.preprocessor_path_config}.")

            return data_arr

        except Exception as e:
            raise CustomException(e, sys)
