import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.neighbors 
from file_checker import loaded_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

'''
Author: Marawan Yakout
Last Edited: 26th Sept 2025
Description:

This script analyzes worldwide GDP trends using two machine learning models and shows difference in Accuracy:

1. K-Nearest Neighbors (KNN) Approx. accuracy: 94.96%
2. Linear Regression (LR)   Approx. accuracy: 87.32%

Note: Code can be improved and can be further used for including the migration Data
'''


# FORMATTING TRAINING DATA & TESTING DATA

#GDP DATA
gdp = loaded_data['organized_data\gdp_rate_worldwide.csv']
gdp_train = gdp[gdp["year"] <= 2022]
gdp_test  = gdp[gdp["year"] >  2022]

#Migration Data
mig = loaded_data['organized_data\migration_oecd.csv']
migration_train = mig[(mig['year'] >= 2012) & (mig['year'] <= 2021)] 
migration_test  = mig[mig['year'] > 2021]

#COMPINED DATA (MIN YEAR 2012 - MAX YEAR 2022)
merged_data = pd.merge(gdp_train, migration_train, on='year') # CEHCK ds_plots.py to see where this is being used

print('\n Unique GDP Data:')
print(merged_data[-5:])

print('\n Training GDP Data:')
print(gdp_train[-5:])


# Training a model to pridect Next 5 years GDP using KNN

X_gdp_train = gdp_train['year'].values.reshape(-1, 1)
y_gdp_train = gdp_train['gdp_worldwide'].values #we want to prdict y

X_gdp_test = gdp_test['year'].values.reshape(-1, 1)
y_gdp_test = gdp_test['gdp_worldwide'].values

model_gdp_knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 4)
model_gdp_knn.fit(X_gdp_train,y_gdp_train)

#ACCURACY TEST

# Test accuracy
if X_gdp_test.size > 0:

    y_true = y_gdp_test
    y_pred_test_knn = model_gdp_knn.predict(X_gdp_test)

   
    mse_knn = mean_squared_error(y_gdp_test, y_pred_test_knn)
    rmse_knn = np.sqrt(mse_knn)
    mae_knn = mean_absolute_error(y_gdp_test, y_pred_test_knn)
    r2_knn = r2_score(y_gdp_test, y_pred_test_knn)

    mape = np.mean(np.abs((y_true - y_pred_test_knn) / y_true)) * 100

    accuracy_pct = 100.0 - mape 


    print("\nKNN Model Accuracy on Test Data:")
    print(f"Mean Squared Error: {mse_knn:.4f}")
    print(f"Mean Absolute Error: {mae_knn:.4f}")
    print(f"R-squared (Regression Score): {r2_knn:.4f}")
    print(f"Approx. accuracy: {accuracy_pct:.2f}%")
else:
    print("\nNo test data available for KNN model evaluation.")

print("Model Parameters:", model_gdp_knn.get_params())

plt.figure(figsize=(20, 8))
plt.scatter(gdp_train['year'],gdp_train['gdp_worldwide'], color='blue', label='Actual Data')


# Actual points
x_range = X_gdp_train.flatten()  # Extend to predict year 2030
y_pred = model_gdp_knn.predict(X_gdp_train)
plt.plot(x_range,y_pred, color='red', linestyle='-', label='KNN fit')

max_year = gdp_train['year'].max() # from 1980 -> 2022
future_years = np.array([max_year + 1, max_year + 2, max_year + 3, max_year + 4, max_year + 5,  max_year + 6, max_year + 7, max_year + 8,]).reshape(-1, 1) #Eight Pridictions

future_gdp = model_gdp_knn.predict(future_years)

# Print KNN predictions Accuracy
print("\nPredicted Real GDP rate Percentage for each future year 2022-2030:")
for year, pred in zip(future_years.flatten(), future_gdp):
    print(f"KNN Predicted GDP rate in {year}: is Aprox {pred:.2f}%")



plt.scatter(future_years, future_gdp, color='green', label='Predicted GDP (Next 5 Years)')
plt.title('World Wide GDP vs Year with KNN Model')
plt.xlabel('Year')
plt.ylabel('Gdp Rate Worldwide (%)')
plt.legend(loc ='upper left')
plt.tight_layout()
plt.grid()
plt.show()



# Training a model to pridect Next 5 years GDP Using Linear Regression

X_gdp_train_lr = gdp_train['year'].values.reshape(-1, 1)
y_gdp_train_lr = gdp_train['gdp_worldwide'].values #we want to prdict y

X_gdp_test_lr = gdp_test['year'].values.reshape(-1, 1)
y_gdp_test_lr = gdp_test['gdp_worldwide'].values

model_gdp_lr = LinearRegression()
model_gdp_lr.fit(X_gdp_train_lr,y_gdp_train_lr)



#ACCURACY TEST

if X_gdp_test.size > 0:

    y_true = y_gdp_test_lr
    y_pred_lr = model_gdp_lr.predict(X_gdp_test_lr)

    mse_lr = mean_squared_error(y_gdp_test_lr, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_gdp_test_lr, y_pred_lr)
    r2_lr = r2_score(y_gdp_test_lr, y_pred_lr)

    mape_lr = np.mean(np.abs((y_true - y_pred_lr) / y_true)) * 100 # MAPE for Linear Regression
    accuracy_pct_lr = 100.0 - mape_lr 

    print("\nLinear Regression Accuracy on Test Data:")
    print(f"Mean Squared Error: {mse_lr:.4f}")
    print(f"Mean Absolute Error: {mae_lr:.4f}")
    print(f"R-squared (Regression Score): {r2_lr:.4f}")
    print(f"Approx. accuracy: {accuracy_pct_lr:.2f}%")

else:
    print("\nNo test data available for Linear Regression model evaluation.")

 
print("Model Parameters:", model_gdp_lr.get_params())
print(f"Slope (GDP change per year): {model_gdp_lr.coef_[0]:.4f}")
print(f"Intercept: {model_gdp_lr.intercept_:.4f}")

# GDP = slope * year + intercept 
# GDP =   -0.0362 * 2023 +  74.7290 = 1.4 EXAMPLE

plt.figure(figsize=(20, 8))
plt.scatter(gdp_train['year'], gdp_train['gdp_worldwide'], color='blue', label='Actual Data')


# Actual points
x_range = gdp_train['year'].values.reshape(-1,1)  # Extend to predict year 2030
y_pred = model_gdp_lr.predict(x_range)
plt.plot(x_range,y_pred, color='red', linestyle='-', label='Regression Line')


max_year = gdp_train['year'].max() # from 1980 -> 2022
future_years_lr = np.array([max_year + 1, max_year + 2, max_year + 3, max_year + 4, max_year + 5,  max_year + 6, max_year + 7, max_year + 8,]).reshape(-1, 1) #Eight Pridictions

future_gdp_lr = model_gdp_lr.predict(future_years_lr)


# Print Linear Regression predictions Accuracy
print("\nAccuracy of Linear Regression - GDP for 2022-2030:")
for year, pred in zip(future_years_lr.flatten(), future_gdp_lr):
    print(f"LR Predicted GDP rate in {year}: is Aprox {pred:.2f}%")


plt.scatter(future_years_lr, future_gdp_lr, color='green', label='Predicted GDP (Next 5 Years)')
plt.title('World Wide GDP vs Year with Linear Regression Model')
plt.xlabel('Year')
plt.ylabel('Gdp Rate Worldwide (%)')
plt.legend(loc ='upper left')
plt.tight_layout()
plt.grid()
plt.show()