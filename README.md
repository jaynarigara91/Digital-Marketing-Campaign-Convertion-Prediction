# Digital Marketing Campaign Conversion Prediction

![Project Presentation](assets/project_presentation.png)

## Project Overview
This project aims to predict the conversion rates of digital marketing campaigns. Using machine learning techniques, the goal is to provide actionable insights for improved campaign targeting, increased conversion rates, and maximized return on advertising spend.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Data Overview](#data-overview)
- [Approach](#approach)
- [Key Results](#key-results)
- [Challenges Faced](#challenges-faced)
- [Technology Stack](#technology-stack)
- [How to Run the Project](#how-to-run-the-project)
- [Future Enhancements](#future-enhancements)

## Features
- Predicts customer conversion based on demographic and engagement metrics.
- Utilizes Exploratory Data Analysis (EDA) to uncover key insights.
- Balances data distribution using SMOTE to handle class imbalance.
- Implements advanced feature engineering for improved model performance.
- Deploys a user-friendly web interface built with Flask.

## Data Overview
- **Dataset:** 8000 rows and 20 features.
- **Target Variable:** `Conversion` (1 = Successful, 0 = Unsuccessful).
- **Data Highlights:**
  - 5 categorical features and 15 numerical features.
  - Data imbalance addressed with SMOTE.
  - Three features (CustomerID, AdvertisingPlatform, AdvertisingTool) were removed due to redundancy.

## Approach
1. **EDA:** Visualized distributions, correlations, and trends to understand the dataset.

   Example Graphs:
   ![Age Distribution](assets/eda_age_distribution.png)
   ![Correlation Heatmap](assets/eda_correlation_heatmap.png)

2. **Feature Engineering:** Created derived metrics like Income per Click, Click-to-Conversion Rate, and Total Interactions.
3. **Model Selection:** Evaluated models such as Logistic Regression, Decision Trees, Random Forest, XGBoost, and K-Nearest Neighbors (KNN).
4. **Model Training:** Used GridSearchCV for hyperparameter tuning.
5. **Deployment:** Built a Flask web app with a clean, user-friendly interface.

## Key Results
- **Best Model:** K-Nearest Neighbors (KNN) with 97% accuracy.
- **Performance Metrics:**
  - AUC: 0.9915
  - Balanced precision and recall across both classes.
- **Observations:** Random Forest and XGBoost also performed well, with accuracies of 95% and 92% respectively.

## Challenges Faced
- **Data Imbalance:**
  - Initial imbalance (7012 conversions vs. 988 non-conversions).
  - Resolved using SMOTE and stratified sampling.
- **Feature Correlations:** High correlations among derived features required careful handling to prevent overfitting.

## Technology Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Flask
- **Dashboard:** Power BI
- **Deployment:** Flask Web Application

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/digital-marketing-conversion-prediction.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Access the app at `http://localhost:5000/`.

## Future Enhancements
- Incorporate more diverse datasets for better generalization.
- Improve the model's performance on imbalanced classes.
- Integrate advanced visualization dashboards using tools like Tableau.

---
Feel free to explore, contribute, or provide feedback!

---
### Acknowledgments
Thanks to the contributors and open-source libraries that made this project possible.
