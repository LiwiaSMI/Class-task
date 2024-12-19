import pandas as pd

df = pd.read_csv('spotify_tracks.csv')

data = df[['popularity', 'duration_ms','acousticness','danceability','energy','instrumentalness']]
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = data.corr()

print(correlation_matrix)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print("-------------------------")


import statsmodels.api as sm

X = data[['popularity']]
y = data['danceability']


X = sm.add_constant(X)

ols_model = sm.OLS(y, X).fit()

print(ols_model.summary())


predictions = ols_model.predict(X)


plt.scatter(X['popularity'], y)
plt.plot(X['popularity'], predictions, color='red', linewidth=2)
plt.title("Simple Linear Regression with Scatter Plot")
plt.xlabel("Popularity")
plt.ylabel("Danceability")
plt.show()


