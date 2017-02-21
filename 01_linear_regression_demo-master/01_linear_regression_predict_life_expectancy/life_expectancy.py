import pandas as pd
from sklearn import linear_model


bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_bmi_values = bmi_life_data[['BMI']]
y_life_values = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(x_bmi_values, y_life_values)

# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)

