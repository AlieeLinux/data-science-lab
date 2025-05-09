import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("student_study_data.csv")

X = df[["StudyHours"]]

y = df["Passed"]

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
conf_matrix = confusion_matrix(y, predictions)
print("Accuracy;", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("\nConfusion Matrix:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
xticklabels=["Predicted Fail", "Predicted Pass"],
yticklabels=["Actual Fail", "Actual Pass"])
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

study_range = np.linspace(0, 6, 100).reshape(-1, 1)
pass_prob = model.predict_proba(study_range)[:, 1]
plt.plot(study_range, pass_prob, color="green")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Predicted Probability of Passing vs Study Hours")

plt.grid(True)
plt.show()
