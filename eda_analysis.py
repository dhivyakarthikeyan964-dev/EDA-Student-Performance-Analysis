import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create charts folder
os.makedirs("charts", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("dataset.csv")

print("\n===== DATASET =====")
print(df)

# 2. Dataset information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

# 3. Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 4. Check duplicate values
print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

# 5. Statistical summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# 6. Create Study Hours distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Study_Hours"], bins=10, kde=True)
plt.title("Distribution of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("charts/study_hours.png")
plt.show()

# 7. Create Attendance distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Attendance"], bins=10, kde=True)
plt.title("Distribution of Attendance")
plt.xlabel("Attendance (%)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("charts/attendance.png")
plt.show()

# 8. Create Final Marks distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Final_Marks"], bins=10, kde=True)
plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("charts/final_marks.png")
plt.show()

# 9. Study Hours vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Marks"
)
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.savefig("charts/study_vs_final.png")
plt.show()

# 10. Previous Marks vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Previous_Marks",
    y="Final_Marks"
)
plt.title("Previous Marks vs Final Marks")
plt.xlabel("Previous Marks")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.savefig("charts/previous_vs_final.png")
plt.show()

# 11. Correlation matrix
correlation = df.corr(numeric_only=True)

print("\n===== CORRELATION MATRIX =====")
print(correlation)

# 12. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png")
plt.show()

print("\n===== EDA COMPLETED SUCCESSFULLY =====")
print("Charts are saved inside the 'charts' folder.")