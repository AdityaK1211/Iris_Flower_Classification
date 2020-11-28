import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Importing the dataset
data = pd.read_csv('Iris.csv')
# Shape
print('Shape:', data.shape)

# Dataset Preview
print(data.head())
# Data Count
print(data.groupby('Species').size())

# Pie chart
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
sizes = [(data['Species'] == 'Iris-setosa').sum(),
         (data['Species'] == 'Iris-versicolor').sum(),
         (data['Species'] == 'Iris-virginica').sum()]
# colors
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
for text in texts:
    text.set_color('grey')
for autotext in autotexts:
    autotext.set_color('grey')
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()

# Split Data
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = data[feature_columns].values
y = data['Species'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Visualizing Data
# Parallel Coordinates
plt.figure(figsize=(10, 5))
parallel_coordinates(data.drop("Id", axis=1), "Species")
plt.title('Parallel Coordinates Plot', fontsize=5, fontweight='bold')
plt.xlabel('Features', fontsize=5)
plt.ylabel('Features values', fontsize=10)
plt.legend(loc=1, prop={'size': 10}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
plt.show()

# Pairplot
plt.figure()
sns.pairplot(data.drop("Id", axis=1), hue="Species", height=3, markers=["o", "s", "D"])
plt.title('Pairplot', fontsize=5, fontweight='bold')
plt.show()

# KNN Classification
classifier = KNeighborsClassifier(n_neighbors=3)
# Fitting the model
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix\n', cm)

# Accuracy
print('\nClassification Report\n', classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred) * 100
print('KNN Accuracy:', str(round(accuracy, 2)) + '%')

# Creating list of K for KNN
k_list = list(range(1, 50, 2))
# Creating list of cv scores
cv_scores = []

# Perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Changing to mis-classification error
MSE = [1 - x for x in cv_scores]
# Finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index=['setosa', 'versicolor', 'virginica'],
                     columns=['setosa', 'versicolor', 'virginica'])
sns.heatmap(cm_df, annot=True)
plt.title('Accuracy using brute:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
