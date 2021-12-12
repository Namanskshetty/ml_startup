import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from scipy.stats import kde
data = pd.read_csv("50_Startups.csv")
data.head(10)

sns.distplot(data['Profit'],bins=5,kde=True)

sns.pairplot(data)

sns.barplot(x='State',y='Profit',data=data, palette="Blues_d")

sns.heatmap(data.corr(), annot=True)

g=sns.FacetGrid(data, col='State')
g=g.map(sns.kdeplot,'Profit')

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data.iloc[:,3]= labelencoder.fit_transform(data.iloc[:,3].values)
print(data['State'].unique())

print(data.head(10))

X = data.iloc[:, :-1].values
Y = data.iloc[:, 4].values

sns.kdeplot(x=data.Administration, y=data.Profit, cmap="Blues", shade=True, thresh=0)
plt.show()

from pandas.plotting import parallel_coordinates
parallel_coordinates(data, 'Profit', colormap=plt.get_cmap("Set2"))
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print('Coefficients: \n', regressor.coef_)
regressor.score(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("Predicted accuray or the socre of the dataset",regressor.score(X_train, Y_train)*100)
print(Y_pred)

df = pd.DataFrame(data={'Predicted value':Y_pred.flatten(),'Actual Value':Y_test.flatten()})
df

df.plot(kind='line')

#https://github.com/namanskshetty

plt.show()
