from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn

df = pd.read_csv("student_data_modeling_medium.csv")
print(df)
#StudentID,Hours_Studied,Attendance,Assignments_Completed,Final_Grade,Passed,Group

x1 = df[["Hours_Studied"]]
x2 = df[["Attendance"]]
x3 = df[["Assignments_Completed"]]


x = pd.concat([x1, x2, x3], axis=1)

y = df["Final_Grade"]

model = LogisticRegression(max_iter=9990)
model.fit(x, y)
predictions = model.predict(x)
confmatrix = confusion_matrix(y, predictions)

seaborn.heatmap(confmatrix)

Accuracy = accuracy_score(y, predictions)

print(Accuracy * 100)

