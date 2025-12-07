🏡 House Price Prediction — Machine Learning Project

Predicting house prices using advanced regression techniques, feature engineering, and hyperparameter tuning.

📌 Project Overview

This project builds a machine learning regression model to predict house prices using the Ames Housing Dataset.

The workflow includes:

✔ Data preprocessing
✔ Exploratory Data Analysis (EDA)
✔ Feature engineering
✔ Outlier handling
✔ Log transformation
✔ Multicollinearity reduction
✔ One-hot encoding
✔ Scaling
✔ Model training (Linear, Ridge, Lasso, ElasticNet)
✔ Hyperparameter tuning (GridSearchCV)
✔ Final model pipeline + saving (.pkl)

The final deployed model is a Tuned Lasso Regression model.

🚀 Tech Stack
Category	Tools Used
Language	Python
Libraries	Pandas, NumPy, Matplotlib, Seaborn
ML Models	Linear Regression, Ridge, Lasso, ElasticNet
Scaling	StandardScaler
Optimization	GridSearchCV
Deployment	Pickle (model saving), Streamlit (optional)
Environment	Google Colab
📂 Project Structure
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── HousePricePrediction.ipynb
├── models/
│   ├── house_price_model.pkl
├── README.md
└── requirements.txt

🛠 1. Data Preprocessing
✔ Handling missing values

Numerical → filled using median

Categorical → filled using mode

✔ Converting column names

Replaced spaces and / to avoid code errors:

df.columns = df.columns.str.replace(" ", "_").str.replace("/", "_")

✔ Outlier detection

Using boxplots → handled using log transform + domain rules.

📊 2. Exploratory Data Analysis (EDA)
✔ Target Distribution

SalePrice was right-skewed, so we applied log transform to stabilize variance.

✔ Correlation Heatmap

Identified strongly correlated features:

Total_Bsmt_SF, Gr_Liv_Area, Garage_Area, Overall_Qual

Also removed multicollinear features like:

Garage_Cars (kept only Garage_Area)

🏗 3. Feature Engineering
✔ New Features Created
df['Total_SF'] = df['Total_Bsmt_SF'] + df['1st_Flr_SF'] + df['2nd_Flr_SF']

df['Total_Bath'] = (
    df['Full_Bath'] +
    df['Half_Bath']*0.5 +
    df['Bsmt_Full_Bath'] +
    df['Bsmt_Half_Bath']*0.5
)

df['House_Age'] = df['Yr_Sold'] - df['Year_Built']
df['Years_Since_Remod'] = df['Yr_Sold'] - df['Year_Remod_Add']

✔ Removing useless / zero-impact features

Order

Mo_Sold

Low_Qual_Fin_SF

Pool_Area

3Ssn_Porch

✔ One-Hot Encoding
cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

🔧 4. Scaling

Scaling was applied after train-test split to avoid data leakage.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

🤖 5. Model Training

Models trained:

Linear Regression

Ridge Regression

Lasso Regression

ElasticNet Regression

🔍 6. Hyperparameter Tuning (GridSearchCV)
Tuned Parameters:
✔ Lasso

alpha = 0.01

✔ Ridge

alpha = 100

✔ ElasticNet

alpha = 0.01, l1_ratio = 0.8

🏆 7. Final Model Performance
Model	R² (Test)	RMSE	MAE
Linear Regression	0.886	0.145	0.080
Ridge Regression	0.889	0.143	0.084
ElasticNet	0.904	0.131	0.078
Lasso Regression (Best)	0.911	0.127	0.077
⭐ Final Selected Model: Tuned Lasso Regression
💾 8. Saving the Model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)

The saved model includes:

✔ StandardScaler
✔ Tuned Lasso Regression
✔ Full preprocessing pipeline

🧪 9. How to Use the Saved Model
import pickle
import numpy as np

model = pickle.load(open("house_price_model.pkl", "rb"))
prediction = model.predict(new_data)

⚙️ 10. Example Prediction Function
def predict_house_price(new_data):
    import pandas as pd
    df = pd.DataFrame([new_data])
    model = pickle.load(open("house_price_model.pkl", "rb"))
    log_pred = model.predict(df)[0]
    return np.exp(log_pred)

🌐 11. Future Improvements

Deploy using Streamlit

Add SHAP explanations

Add cross-validation visualization

Automate preprocessing pipeline with ColumnTransformer

Try advanced models (XGBoost, LightGBM, CatBoost)

🧑‍💻 Author

Harshit Kumar Pandey
B.Tech CSE (IoT Specialization)
ML & AI Enthusiast