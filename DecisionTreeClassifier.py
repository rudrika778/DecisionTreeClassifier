#import required libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#Load the iris dataset
iris=load_iris()
x=iris.data  #Features : sepal length; sepal width ; petal length ; petal width
y=iris.target #Labels: 0=setosa ; 1=versicolor ;2=virginica

#Split the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Create and train the decision tree classifier
model=DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(x_train,y_train)

#Predict on the test set
y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
             xticklabels=iris.target_names,
             yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Evaluate the model
print("Classification REPORT:\n")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

#Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.title("Decision Tree Classifier")
plt.show()

imp=model.feature_importances_
features_names=iris.feature_names
plt.barh(features_names,imp,color="steelblue")
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.grid(True)
plt.show()

#new prediction
new_flower=[[5.1,3.2,1.4,0.2]]
prediction=model.predict(new_flower)[0]
print(f"Predicted species:{iris.target_names[prediction]}")
