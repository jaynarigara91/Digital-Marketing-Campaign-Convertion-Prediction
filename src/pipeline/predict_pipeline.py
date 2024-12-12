import os
import sys
import pandas as pd
from flask import Flask, request, render_template
from src.exception import CustomException
from src.utils import load_object




class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,feature):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocess_path = os.path.join('artifacts','preprocessor.pkl')
            print('Before Loading...')
            model = load_object(model_path)
            preprocessor = load_object(preprocess_path)
            print('After Loading...')
            data_scaled = preprocessor.transform(feature)
            prediction = model.predict(data_scaled)
            if prediction==1:
                return 'Conversion'
            elif prediction==0:
                return 'Not Conversion'
        
        except Exception as e:
            raise CustomException(e,sys)


# CustomData class for handling input with additional features
class CustomData:
    def __init__(self, Age: int, Gender: str, Income: float, CampaignChannel: str,
                 CampaignType: str, AdSpend: float, ClickThroughRate: float,
                 ConversionRate: float, WebsiteVisits: int, PagesPerVisit: float,
                 TimeOnSite: float, SocialShares: int, EmailOpens: int,
                 EmailClicks: int, PreviousPurchases: int, LoyaltyPoints: int):
        """
        Initialize CustomData with user-provided input attributes.
        """
        self.Age = Age
        self.Gender = Gender
        self.Income = Income
        self.CampaignChannel = CampaignChannel
        self.CampaignType = CampaignType
        self.AdSpend = AdSpend
        self.ClickThroughRate = ClickThroughRate
        self.ConversionRate = ConversionRate
        self.WebsiteVisits = WebsiteVisits
        self.PagesPerVisit = PagesPerVisit
        self.TimeOnSite = TimeOnSite
        self.SocialShares = SocialShares
        self.EmailOpens = EmailOpens
        self.EmailClicks = EmailClicks
        self.PreviousPurchases = PreviousPurchases
        self.LoyaltyPoints = LoyaltyPoints

    def to_dataframe(self):
        """
        Convert the user input data into a pandas DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the user input data.
        """
        try:
            input_data = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Income": [self.Income],
                "CampaignChannel": [self.CampaignChannel],
                "CampaignType": [self.CampaignType],
                "AdSpend": [self.AdSpend],
                "ClickThroughRate": [self.ClickThroughRate],
                "ConversionRate": [self.ConversionRate],
                "WebsiteVisits": [self.WebsiteVisits],
                "PagesPerVisit": [self.PagesPerVisit],
                "TimeOnSite": [self.TimeOnSite],
                "SocialShares": [self.SocialShares],
                "EmailOpens": [self.EmailOpens],
                "EmailClicks": [self.EmailClicks],
                "PreviousPurchases": [self.PreviousPurchases],
                "LoyaltyPoints": [self.LoyaltyPoints],
            }
            data = pd.DataFrame(input_data)
            
            # Perform calculations
            data['EmailEngagement'] = data['EmailOpens'] + data['EmailClicks']
            data['SiteEngagement'] = data['WebsiteVisits'] * data['PagesPerVisit'] * data['TimeOnSite']
            data['IncomePerClick'] = data['Income'] / (data['ClickThroughRate']+1)  
            data['AdSpendPerClick'] = data['AdSpend'] / (data['ClickThroughRate']+1)  
            data['ClickToConversionRate'] = data['ConversionRate'] / (data['ClickThroughRate']+1) 
            data['TotalInteractions'] = data['WebsiteVisits'] + data['EmailOpens'] + data['EmailClicks'] + data['SocialShares'] + data['PreviousPurchases']
            
            return data

        except Exception as e:
            raise CustomException(e, sys)




