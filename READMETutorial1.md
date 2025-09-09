# Introduction to Machine Learning Concepts with Python

This repository contains lab materials for **BICT 332: Artificial Intelligence** at the **University of Mpumalanga**.  
Prepared by *Dr. Ayorinde Olanipekun*.

## Lab Overview
**Title:** Introduction to Machine Learning Concepts with Python  
**Chapter:** Giving Computers the Ability to Learn from Data  
**Duration:** ~90 minutes  

The lab provides a gentle, hands-on introduction to machine learning concepts while setting up the environment for future chapters.

---

## Objectives
- Understand the basic types of machine learning:
  - Supervised  
  - Unsupervised  
  - Reinforcement  
- Explore the **machine learning pipeline**  
- Work with simple datasets to observe ML in action  
- Gain hands-on experience with **Python tools** for machine learning  

---

## Lab Exercises

### Exercise 1: Setting Up the Environment (15 min)
- Verify Python installation  
- Install required packages:
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```
- Test installations:
  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn import datasets

  print("All packages imported successfully!")
  ```

---

### Exercise 2: Exploring Types of Learning (20 min)

#### Part A: Supervised Learning – Iris Classification
- Load and explore the Iris dataset  
- Train a **Logistic Regression** model  
- Evaluate accuracy & classification report  

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
```

#### Part B: Unsupervised Learning – Iris Clustering
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-means Clustering Results")
plt.show()
```

---

### Exercise 3: The Machine Learning Pipeline (25 min)

1. **Data Preparation**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

2. **Model Training**
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   model = KNeighborsClassifier(n_neighbors=3)
   model.fit(X_train, y_train)
   ```

3. **Model Evaluation**
   ```python
   from sklearn.metrics import accuracy_score
   y_pred = model.predict(X_test)
   print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
   ```

---

### Exercise 4: Reflection Questions (15 min)
- What differences did you observe between supervised and unsupervised learning?  
- How did the train-test split help in evaluating the model?  
- What factors might affect model accuracy?  
- Can you think of real-world applications for each type of learning?  

---

## Wrap-up
- Reviewed ML concepts & hands-on exercises  
- Discussed extensions (try different models, parameters)  
- Shared resources for further exploration  

---

## Assessment
- Completion of all exercises  
- Thoughtful reflection responses  
- Demonstration of understanding  

---

## Dataset
- [Iris Dataset on Kaggle](https://www.kaggle.com/datasets/vikrishnan/iris-dataset)
