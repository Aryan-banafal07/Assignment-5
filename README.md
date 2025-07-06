# 🏠 House Prices - Advanced Regression Techniques
This project solves the Kaggle regression challenge by predicting final house sale prices using advanced preprocessing, feature engineering, and regression models like Lasso and Random Forest.

# 📂 Dataset
Kaggle Competition: House Prices - Advanced Regression Techniques

Files used:

train.csv

test.csv

sample_submission.csv

# 🧠 Techniques Used
Data Cleaning (missing value imputation)

Feature Engineering:

Total Square Footage (TotalSF)

Total Bathrooms (TotalBath)

House Age and Remodelling Age

One-hot Encoding for categorical features

Regression Models:

LassoCV for regularized linear regression

Random Forest Regressor for ensemble tree learning

Model Evaluation using RMSE on log-transformed SalePrice

# 📊 Visualizations
Distribution of SalePrice (original & log-transformed)

Model residuals (Lasso)

Feature importance:

Lasso non-zero coefficients

Top 30 features from Random Forest

📈 Model Performance (on training set)
Model	RMSE (log scale)
LassoCV	~0.12–0.15
Random Forest	~0.06–0.09

⚠️ Note: These are training RMSEs. For reliable generalization, cross-validation or test leaderboard is advised.

📁 File Structure
Copy
Edit
├── train.csv
├── test.csv
├── submission.csv
├── house_price_regression.ipynb
└── README.md
🚀 How to Run
Clone the repository or open in Google Colab

Upload train.csv and test.csv

Run all cells to:

Preprocess data

Train Lasso and Random Forest

Visualize insights

Generate submission.csv

🛠️ Libraries Used
pandas

numpy

seaborn

matplotlib

scikit-learn

📌 Future Improvements
Use advanced models like XGBoost, LightGBM, or Stacking

Perform hyperparameter tuning

Add cross-validation RMSE

Experiment with feature selection techniques
