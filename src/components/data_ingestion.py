import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_traine import ModelTrainer
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataIngestionConfig:
    """
        setting train_path ,test_path, data_path
    """
    train_path_config : str = os.path.join('artifacts','train.csv')
    test_path_config: str = os.path.join('artifacts','test.csv')
    Rawdata_path_config: str = os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        # Initialize the data ingestion process with configuration paths
        self.data_config_path = DataIngestionConfig()
        
    def initialize_data_ingestion(self):
        """
        This method handles the data ingestion process:
        - Loads the dataset.
        - Feature Engineering
        - Balances the classes by sampling.
        - Saves the processed data into train and test datasets.
        """
        try:
            
            data = pd.read_csv('Notebook\Data\digital_marketing_campaign_dataset.csv')
            logging.info('Data Load From Folder Successfully.')
            
            data.drop(columns=['CustomerID','AdvertisingPlatform','AdvertisingTool'],inplace=True)  # Removing Un wanted features
            
            data['EmailEngagement'] = data['EmailOpens'] + data['EmailClicks']
            data['SiteEngagement'] = data['WebsiteVisits'] * data['PagesPerVisit'] * data['TimeOnSite']
            data['IncomePerClick'] = data['Income'] / (data['ClickThroughRate'] + 1)  
            data['AdSpendPerClick'] = data['AdSpend'] / (data['ClickThroughRate'] + 1)  
            data['ClickToConversionRate'] = data['ConversionRate'] / (data['ClickThroughRate'] + 1) 
            data['TotalInteractions'] = data['WebsiteVisits'] + data['EmailOpens'] + data['EmailClicks'] + data['SocialShares'] + data['PreviousPurchases']
            
            logging.info('Feature Engineering completed')
            
            class_1_data = data[data['Conversion'] == 1]
            
            # stratified data sampling
            sampled_class_1_data, _ = train_test_split(
                class_1_data, 
                train_size=988, 
                stratify=class_1_data['Conversion'], 
                random_state=42
            )
            
            # Combine the sampled class_1_data with the data where Conversion = 0
            result = pd.concat([sampled_class_1_data, data[data['Conversion'] == 0]])
            
            logging.info("Data Balance with 988 raw by stratified sampling.")
            
            # Shuffle the resulting dataset and reset the index
            data = result.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # make folder named artifacts
            os.makedirs(os.path.dirname(self.data_config_path.Rawdata_path_config), exist_ok=True)
            
            logging.info("artifacts folder created.")
        
            data.to_csv(self.data_config_path.Rawdata_path_config, index=False, header=True)
            
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(data, test_size=0.8, random_state=42)
            
            logging.info("Splited Data Into Train And test.")
            
            train_set.to_csv(self.data_config_path.train_path_config, index=False, header=True)
           
            test_set.to_csv(self.data_config_path.test_path_config, index=False, header=True)
            
            logging.info('Saved train_data, test_data and Raw_data into artifacts folder.')

            return (
                self.data_config_path.train_path_config,  #train_path
                self.data_config_path.test_path_config, #test_path
                self.data_config_path.Rawdata_path_config
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
