"""
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
Author: Rinae Mathaulula

Classifies Iris flowers (setosa, versicolor, virginica) from four measurements
(sepal length/width, petal length/width) and compares several classifiers.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# 1. Load data -------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print("Shape:", df.shape)
print(df.head(), "\n")
print("Missing values per column:\n", df.isna().sum(), "\n")
print("Class balance:\n", df["species"].value_counts(), "\n")
print(df.groupby("species")[iris.feature_names].mean().round(2), "\n")

# 2. Exploratory plots ------------------------------------------------------
pair = sns.pairplot(df.drop(columns="target"), hue="species", corner=True)
pair.figure.suptitle("Iris features by species", y=1.02)
pair.savefig("pairplot.png", dpi=120, bbox_inches="tight")
plt.close("all")

plt.figure(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature correlation")
plt.tight_layout(); plt.savefig("correlation.png", dpi=120); plt.close()

# 3. Train/test split -------------------------------------------------------
X, y = df[iris.feature_names], df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# 4. Compare models with 5-fold cross-validation ---------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    rows.append((name, cv_scores.mean(), cv_scores.std(), test_acc))
results = pd.DataFrame(rows, columns=["Model", "CV mean acc", "CV std", "Test acc"]).sort_values("CV mean acc", ascending=False)
print(results.round(3).to_string(index=False), "\n")
results.round(4).to_csv("model_comparison.csv", index=False)

# 5. Evaluate the best model in detail -------------------------------------
best_name = results.iloc[0]["Model"]
best = models[best_name]
y_pred = best.predict(X_test)
print(f"Best model: {best_name}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=iris.target_names).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

# Feature importance from the random forest (which measurements matter most)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values()
imp.plot.barh(color="#4C72B0", figsize=(6, 4))
plt.title("Random Forest feature importance"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120); plt.close()

# 6. Predict a new flower ---------------------------------------------------
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]], columns=iris.feature_names)
for row, pred in zip(sample.values, best.predict(sample)):
    print(f"Measurements {row} -> predicted species: {iris.target_names[pred]}")
