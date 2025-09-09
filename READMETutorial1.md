# Introduction to Machine Learning Concepts with Python

**University of Mpumalanga**  
**BICT 332: Artificial Intelligence**  
*By Dr. Ayorinde Olanipekun*  

## Lab Plan – Chapter 1  
**Title:** Giving Computers the Ability to Learn from Data  

---

## Objectives
1. Understand the basic types of machine learning (supervised, unsupervised, reinforcement)  
2. Explore the machine learning pipeline  
3. Work with simple datasets to observe machine learning in action  
4. Gain hands-on experience with basic Python tools for machine learning  

---

## Lab Exercises

### Exercise 1: Setting Up the Environment (15 minutes)
1. Verify Python installation  
2. Install required packages:  
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
3. Test installations:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn import datasets

   print("All packages imported successfully!")
   ```

---

### Exercise 2: Exploring Different Types of Learning (20 minutes)

#### Part A: Supervised Learning – Iris Classification
```python
# Load iris dataset
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Explore the dataset
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 samples:\n", X[:5])

# Simple visualization
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset")
plt.show()
```

Another approach:
```python
# Step 1: Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 2: Load the Iris Dataset
iris = load_iris()
X = iris.data              # Features: sepal length, width, petal length, width
y = iris.target            # Target: 0 - Setosa, 1 - Versicolour, 2 - Virginica

# Optional: Show the first few samples
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("Sample Data:")
print(df.head())

# Step 3: Split the Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

#### Part B: Unsupervised Learning – Iris Clustering
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Compare with true labels
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-means Clustering Results")
plt.show()
```

---

### Exercise 3: The Machine Learning Pipeline (25 minutes)

#### Step 1: Data Preparation
```python
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
```

#### Step 2: Model Training
```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```

#### Step 3: Model Evaluation
```python
# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
```

---

### Exercise 4: Reflection Questions (15 minutes)
1. What differences did you observe between supervised and unsupervised learning approaches?  
2. How did the train-test split help in evaluating the model?  
3. What factors might affect the accuracy of the model?  
4. Can you think of real-world applications for each type of learning?  

---

## Lab Wrap-up
- Review key concepts covered  
- Discuss potential extensions (e.g., trying different models, parameters)  
- Provide resources for further exploration  

---

## Assessment
- Completion of all code exercises  
- Thoughtful responses to reflection questions  
- Demonstration of understanding during discussion  

---

## Dataset
- [Iris Dataset on Kaggle](https://www.kaggle.com/datasets/vikrishnan/iris-dataset)  

---

⚡ This lab provides a hands-on introduction to machine learning concepts while setting up the environment for future chapters. Designed for completion within a **90-minute session** with instructor support.  
