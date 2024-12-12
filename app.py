from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Age=int(request.form.get("Age")),
            Gender=request.form.get('Gender'),
            Income=int(request.form.get('Income')),
            CampaignChannel=request.form.get('CampaignChannel'),
            CampaignType=request.form.get('CampaignType'),
            AdSpend	=float(request.form.get('AdSpend')),
            ClickThroughRate=float(request.form.get('ClickThroughRate')),
            ConversionRate=float(request.form.get('ConversionRate')),
            WebsiteVisits=int(request.form.get('WebsiteVisits')),
            PagesPerVisit=float(request.form.get('PagesPerVisit')),
            TimeOnSite=float(request.form.get('TimeOnSite')),
            SocialShares=int(request.form.get('SocialShares')),
            EmailOpens=int(request.form.get('EmailOpens')),
            EmailClicks=int(request.form.get('EmailClicks')),
            PreviousPurchases=int(request.form.get('PreviousPurchases')),
            LoyaltyPoints=int(request.form.get('LoyaltyPoints'))

        )
        
        pred_df = data.to_dataframe()
        print(pred_df)
        print('Before Prediction')
        
        Predict_pipeline = PredictPipeline()
        
        print('Mid Prediction')
        result = Predict_pipeline.predict(pred_df)
        print('After Prediction')
        
        return render_template('home.html',results=result)
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  
    
## Search this in browser After run 'python app.py'

# search : http://localhost:5000