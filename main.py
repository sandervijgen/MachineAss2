import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

data_set = pd.read_csv("winequality-red.csv")

#Get an idea of how the data looks
print(data_set.head())  # Display the first few rows of the dataset
print(data_set.info())  # Check data types and missing values
print(data_set.isnull().sum())
print(data_set.describe())  # Summary statistics

# Correlation heatmap
numeric_data = data_set.select_dtypes(include='number')  # Select only numeric columns
cor = numeric_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdYlBu)
plt.show()

# Define features and target variable, Only features with a positive correlation with the target
X = data_set[['citric acid','residual sugar','free sulfur dioxide', 'pH','sulphates','alcohol']]
y = data_set.quality

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#classification with tree
treeModel = tree.DecisionTreeClassifier(max_depth = 2, criterion = "entropy")
treeModel = treeModel.fit(X_train, y_train)

#predict test values for rating tree model
y_pred = treeModel.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(y_test)
print(y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")

feature_names = ['citric acid','residual sugar','free sulfur dioxide', 'pH','sulphates','alcohol']
class_names = ['3','4','5','6','7','8']



fig = plt.figure(figsize=(25,20))
plot = tree.plot_tree(treeModel,
               feature_names=feature_names,
               class_names=class_names,
               filled=True)
plt.show()


