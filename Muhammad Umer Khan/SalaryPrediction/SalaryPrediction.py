import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("C:/Users/student/Desktop/Salary_Data.csv")
print(data.head())

print(data.isnull().sum())

figure = px.scatter(data_frame = data, 
                    x="Salary",
                    y="YearsExperience", 
                    size="YearsExperience", 
                    trendline="ols")
figure.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = np.asanyarray(data[["YearsExperience"]])
y = np.asanyarray(data[["Salary"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)

a = float(input("Years of Experience : "))
features = np.array([[a]])
print("Predicted Salary = ", model.predict(features))

    