# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Step 2: Load the dataset
df = pd.read_csv("bank-additional-full.csv", sep=';')
df = df.head(10000)  # Only use the first 10,000 rows

# Step 3: Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 4: Features and target variable
X = df.drop("y", axis=1)  # Features
y = df["y"]               # Target

# Step 5: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Visualize the tree (Optional but useful for your report)
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=label_encoders['y'].classes_, filled=True, max_depth=3)
plt.title("Decision Tree (first 3 levels)")
plt.show()
