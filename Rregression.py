import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['Id']

train['is_train'] = 1
test['is_train'] = 0
test['SalePrice'] = None
full = pd.concat([train, test], axis=0)

cat_cols = full.select_dtypes(include='object').columns
for col in cat_cols:
    full[col] = full[col].fillna('Missing')

full['LotFrontage'] = full.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.mean()))
for col in full.select_dtypes(include=['float64', 'int64']):
    full[col] = full[col].fillna(full[col].mean())

full['Age'] = full['YrSold'] - full['YearBuilt']
full['RemodAge'] = full['YrSold'] - full['YearRemodAdd']
full['TotalBath'] = (full['FullBath'] + 0.5*full['HalfBath'] +
                     full['BsmtFullBath'] + 0.5*full['BsmtHalfBath'])
full['TotalSF'] = (full['1stFlrSF'] + full['2ndFlrSF'] + full['TotalBsmtSF'])

full = pd.get_dummies(full, drop_first=True)

target = train['SalePrice']
saleprice_cols = [col for col in full.columns if col.startswith('SalePrice_')]
full = full.drop(columns=saleprice_cols)

train_df = full[full['is_train'] == 1].drop(['is_train', 'Id'], axis=1)
test_df = full[full['is_train'] == 0].drop(['is_train', 'Id'], axis=1)

X_train = train_df
y_train = np.log1p(target)

# Lasso Regression
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_train)
lasso_rmse = np.sqrt(mean_squared_error(y_train, lasso_pred))
print(f"Lasso RMSE (log scale): {lasso_rmse:.4f}")

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_train)
rf_rmse = np.sqrt(mean_squared_error(y_train, rf_pred))
print(f"Random Forest RMSE (log scale): {rf_rmse:.4f}")

# Predict with better model (here: Lasso)
final_preds = np.expm1(lasso_model.predict(test_df))
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_preds
})
submission.to_csv('submission.csv', index=False)

# Visualizations
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(target, kde=True, bins=50)
plt.title('Original SalePrice Distribution')

plt.subplot(1, 2, 2)
sns.histplot(y_train, kde=True, bins=50, color='orange')
plt.title('Log-transformed SalePrice Distribution')
plt.tight_layout()
plt.show()

residuals = y_train - lasso_pred
plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True, bins=40, color='green')
plt.title("Lasso Residuals (Log SalePrice)")
plt.xlabel("Residuals")
plt.show()

lasso_coefs = pd.Series(lasso_model.coef_, index=X_train.columns)
imp_lasso = lasso_coefs[lasso_coefs != 0].sort_values()

plt.figure(figsize=(10, 12))
imp_lasso.plot(kind='barh')
plt.title("Lasso Feature Importance (Non-zero Coefficients)")
plt.tight_layout()
plt.show()

rf_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_rf_features = rf_importances.sort_values(ascending=False).head(30)

plt.figure(figsize=(10, 12))
top_rf_features.sort_values().plot(kind='barh')
plt.title("Random Forest Top 30 Feature Importances")
plt.tight_layout()
plt.show()
