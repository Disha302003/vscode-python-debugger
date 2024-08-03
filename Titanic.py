import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
titanic_df = sns.load_dataset('titanic')

# Fill missing values
titanic_df['age'].fillna(titanic_df['age'].median(), inplace=True)
titanic_df['embarked'].fillna(titanic_df['embarked'].mode()[0], inplace=True)
titanic_df['deck'].fillna('Unknown', inplace=True)

# One-hot encoding for categorical variables
titanic_df = pd.get_dummies(titanic_df, columns=['sex', 'embarked', 'pclass'], drop_first=True)

# Select relevant features
features = ['age', 'fare', 'sibsp', 'parch', 'sex_male', 'embarked_Q', 'embarked_S', 'pclass_2', 'pclass_3']
X = titanic_df[features]
y = titanic_df['survived']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(classification_report(y_test, rf_y_pred))

# Train and evaluate Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_y_pred)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}")
print(classification_report(y_test, gb_y_pred))
