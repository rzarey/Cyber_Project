import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from thundersvm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data-train.csv")

data = data.drop(['Timestamp', 'Payload Data', 'User Information', 'Device Information',
                  'Network Segment', 'Geo-location Data', 'Log Source'], axis=1)

categorical_features = ['Source IP Address', 'Destination IP Address', 'Protocol', 'Packet Type',
                        'Traffic Type', 'Attack Type', 'Attack Signature', 'Action Taken']

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_features = encoder.fit_transform(data[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

numerical_features = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores']
data = data.drop(categorical_features, axis=1)
data = pd.concat([data, encoded_df], axis=1)

X = data.drop('Severity Level', axis=1)
y = data['Severity Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

svm_model = SVC(kernel='rbf')  # Use RBF kernel as a starting point

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_svm_model.classes_, yticklabels=best_svm_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()