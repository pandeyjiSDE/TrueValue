🏡 House Price Prediction — ML Regression Project

Predicting house prices using regression models, feature engineering, and hyperparameter tuning on the Ames Housing dataset.

🚀 Project Summary

This project builds a machine learning pipeline to predict house prices.
Key steps include preprocessing, EDA, feature engineering, scaling, model training, hyperparameter tuning, and final deployment (saved model pipeline).

Final selected model → Tuned Lasso Regression

🔧 Tech Used

Python, Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (Linear, Ridge, Lasso, ElasticNet, Scaling, GridSearchCV)
Google Colab, Jupyter Notebook

📁 Project Structure
TrueValue/
│── data/ (AmesHousing.csv)
│── models/ (house_price_model.pkl)
│── notebook/ (analysis + training notebook)
│── requirements.txt
│── README.md

🧹 Data Preprocessing

Handled missing numerical values using median

Handled missing categorical values using mode

Cleaned column names (removed spaces/slashes)

Removed outliers and zero-impact columns

Applied log transform on SalePrice due to heavy right skew

Fixed multicollinearity (removed correlated features like Garage_Cars, BsmtFin_SF_1, TotRms_AbvGrd)

⚙️ Feature Engineering

Total_SF = Total basement + 1st + 2nd floor area

Total_Bath = Full + Half + Basement baths

House_Age = Yr_Sold – Year_Built

Years_Since_Remod = Yr_Sold – Year_Remod_Add

Categorical → Converted using One-Hot Encoding
Scaling → StandardScaler (fit on train only to prevent leakage)

🤖 Models Trained

Linear Regression

Ridge Regression

Lasso Regression

ElasticNet Regression

After tuning with GridSearchCV:

Lasso performed best (alpha = 0.01)

📈 Final Model Performance (Concise)

Lasso achieved the best R² and lowest error

Final model saved as house_price_model.pkl

📦 Load & Predict
import pickle
import pandas as pd

model = pickle.load(open("house_price_model.pkl", "rb"))

def predict_price(data):
    df = pd.DataFrame([data])
    return model.predict(df)[0]

🚧 Future Work

Deploy using Streamlit

Add SHAP explainability

Try advanced models (XGBoost, LightGBM, CatBoost)

Add API (Flask/FastAPI)

👨‍💻 Author

Harshit Kumar Pandey
B.Tech CSE (IoT) | ML & AI Enthusiast