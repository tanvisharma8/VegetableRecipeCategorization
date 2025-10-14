# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 14:40:57 2025

@author: admin
"""

# Improved Vegetable Recipe Categorization using Naive Bayes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\admin\Downloads\vegetable_recipes.csv")

# Step 2: Data Cleaning
df['Ingredients'] = df['Ingredients'].str.lower()
df['Main Vegetable'] = df['Main Vegetable'].str.lower()

# Combine Ingredients + Main Vegetable as text feature
df['Text_Feature'] = df['Ingredients'] + " " + df['Main Vegetable']

print("Dataset Preview:")
print(df.head())

# Step 3: Visualize Data
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Cuisine", order=df['Cuisine'].value_counts().index)
plt.title("Number of Recipes per Cuisine")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Cooking Method", order=df['Cooking Method'].value_counts().index)
plt.title("Number of Recipes per Cooking Method")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Main Vegetable", order=df['Main Vegetable'].value_counts().index)
plt.title("Distribution of Main Vegetables")
plt.xticks(rotation=45)
plt.show()

# Step 4: Preprocessing
X = df['Text_Feature']   # Features (Ingredients + Main Vegetable)
y = df['Cuisine']        # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Complement Naive Bayes Model
model = ComplementNB(alpha=0.5)  # Better for imbalanced data
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Step 7: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred, labels=df['Cuisine'].unique())
sns.heatmap(cm, annot=True, fmt="d", xticklabels=df['Cuisine'].unique(), yticklabels=df['Cuisine'].unique(), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 8: Predict New Recipe
new_recipe = ["Broccoli, Carrot, Bell Pepper, Garlic, Olive Oil, Soy Sauce"]
new_text = [r.lower() + " broccoli" for r in new_recipe]  # Include main vegetable in feature
new_vec = vectorizer.transform(new_text)
predicted_cuisine = model.predict(new_vec)
print(f"Predicted Cuisine for '{new_recipe[0]}' is: {predicted_cuisine[0]}")
