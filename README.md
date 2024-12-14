# Digital Marketing Campaign Conversion Prediction

## Project Overview
This project aims to predict the conversion rates of digital marketing campaigns. Using machine learning techniques, the goal is to provide actionable insights for improved campaign targeting, increased conversion rates, and maximized return on advertising spend.

![Screenshot 2024-12-11 191617](https://github.com/user-attachments/assets/cf4fcf96-255b-423d-a576-cc6168a56c31)
![Screenshot 2024-12-11 191813](https://github.com/user-attachments/assets/702eaab8-4e83-49c1-bd5f-6b88d289b0e3)
![Screenshot 2024-12-11 191830](https://github.com/user-attachments/assets/5965d53d-19b5-4e90-9aab-a978b2a2e4b8)

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
   ![Age Distribution](https://github.com/user-attachments/assets/962ce9ce-0145-43a3-b14a-3ff8e04c76dc)
   ![Correlation Heatmap](https://github.com/user-attachments/assets/c61ba5be-9d13-40b0-b36f-0d8b718ad45a)


2. **Feature Engineering:** Created derived metrics like Income per Click, Click-to-Conversion Rate, and Total Interactions and 3 more.
3. **Model Selection:** Evaluated models such as Logistic Regression, Decision Trees, Random Forest, XGBoost, and K-Nearest Neighbors (KNN).
4. **Model Training:** Used GridSearchCV for hyperparameter tuning.
5. **Deployment:** Built a Flask web app with a clean, user-friendly interface.

## Key Results
- **Best Model:** K-Nearest Neighbors (KNN) with 97% accuracy.
- **Performance Metrics:**
  - AUC: 0.9915
  - Balanced precision and recall across both classes.
- **Observations:** Random Forest and XGBoost also performed well, with accuracies of 95% and 92% respectively.
- **Confusion Metrics Of K-Nearest Neihbors (KNN) :**
<div style="display: flex; flex-direction: row; gap: 20px;">
    <img src="https://github.com/user-attachments/assets/c744ce99-efd7-41b7-8abd-cde123e188e0" alt="Confusion Matrix" width="400">
    <img src="https://github.com/user-attachments/assets/724adc30-c62f-4323-9821-f8121a282ba7" alt="Screenshot 2024-12-10 081754" width="400">
</div>

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

## Poer BI DeshBoard
![Screenshot 2024-12-11 171937](https://github.com/user-attachments/assets/1c566711-f9bb-4d21-837b-03738fa7c21d)

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
