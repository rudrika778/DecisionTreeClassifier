# 🌳 Decision Tree Classifier using Iris Dataset

## 📌 Overview

This project demonstrates the implementation of a **Decision Tree Classifier** using the **Iris Dataset** from Scikit-Learn.

The objective was to understand how Decision Trees make predictions and explore concepts like **Gini Impurity**, **Feature Importance**, **Train-Test Split**, and **Overfitting**.

---

## 🌸 Dataset

The Iris dataset contains:

* 150 flower samples
* 3 species: Setosa, Versicolor, Virginica
* 4 features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width

---

## 🚀 Features

* Decision Tree Classification
* Train-Test Split (70% Train, 30% Test)
* Decision Tree Visualization
* Confusion Matrix Heatmap
* Feature Importance Analysis
* Prediction on Unseen Data

---

## 📊 Key Findings

* The model uses **Gini Impurity** to find the best splits.
* Limiting the tree depth (`max_depth=3`) reduced overfitting while maintaining accuracy.
* **Petal Length** was the most important feature, contributing around **92%** of the decisions.

---

## 🛠️ Technologies Used

* Python
* Scikit-Learn
* Pandas
* Matplotlib
* Seaborn

---

## ▶️ Run the Project

```bash
git clone https://github.com/rudrika778/DecisionTreeClassifier.git
cd DecisionTreeClassifier
pip install -r requirements.txt
python decision_tree_classifier.py
```

---


Learning Machine Learning one project at a time 🚀
