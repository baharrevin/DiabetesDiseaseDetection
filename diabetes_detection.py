
# ******** linear regrasyon  ********

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
df['target'] = diabetes.target
df.to_csv('diabetes_data.csv', index=False)


# ***** 1) basit linear regrasyon : Bağımsız değişken: BMI **

x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# numpy array formatında geliyor. 
# Bu ham veriye başlık (kolon isimleri) eklemek ve daha okunabilir bir hale getirmek için pandas.DataFrame yapısına dönüştürüyoruz.
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

# önemli kısım !! bmi'yi araştırmak istediğimiz için x için bmi'yi aldık
x_bmi_train =  x_train[["bmi"]]
x_bmi_test =  x_test[["bmi"]]

model_simple = LinearRegression()
model_simple.fit(x_bmi_train, y_train)    # eğitim

pred_simple = model_simple.predict(x_bmi_test)

r2_simple = r2_score(y_test, pred_simple)
mae_simple = mean_absolute_error(y_test, pred_simple)
mse_simple = mean_squared_error(y_test, pred_simple)


# ***** 2) çoklu linear regrasyon : Bağımsız değişken: BMI **

x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

model_multiple = LinearRegression()
model_multiple.fit(x_train,y_train)

pred_multiple = model_multiple.predict(x_test)

r2_multiple = r2_score(y_test, pred_multiple)
mae_multiple = mean_absolute_error(y_test, pred_multiple)
mse_multiple = mean_squared_error(y_test, pred_multiple)



# ***** interpretation *****

print("for simple linear regression; ")
print("\nr^2 value for simple linear regression: ", r2_simple)
print("\nMean absolute error for simple linear regression: ", mae_simple)
print("\nMean square error for simple linear regression: ", mse_simple)

print("for multiple linear regression; ")
print("\nr^2 value for multiple linear regression: ", r2_multiple)
print("\nMean absolute error for multiple linear regression: ", mae_multiple)
print("\nMean square error for multiple linear regression: ", mse_multiple)


"""
r^2 for simple linear regression
23% explained part: If BMI is high, disease score is usually high → model understands this part. 
77% unexplained part: But in some people BMI is high but score is low → model cannot explain this because it does not know other factors.
"""

print("\nFor r^2 values of simple and multiple linear regressions;")
print(f"simple linear regression: {r2_simple} and multiple linear regression: {r2_multiple}")
print("(multiple > simple) It shows that The multiple linear regression model explained the target better because it used more variables. This is an expected result. ")

print("\nFor Mean absolute error values of simple and multiple linear regressions;")
print(f"simple linear regression: {mae_simple} and multiple linear regression: {mae_multiple}")
print("Mean absolute error of the multiple linear regression model is smaller than the simple linear regression. It means that the multiple linear regression model makes more accurate predictions.")

print("\nFor Mean squared error values of simple and multiple linear regressions;")
print(f"simple linear regression: {mse_simple} and multiple linear regression: {mse_multiple}")
print("Mean squared error of the multiple linear regression model is smaller than the simple linear regression. It means that the multiple linear regression model makes more accurate predictions.")