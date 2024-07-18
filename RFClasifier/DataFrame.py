import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Load Data
data_path = 'D:\Kuliah\CYBER_PROJECT\data-train.csv'  # Ganti dengan path file CSV Anda
data = pd.read_csv(data_path, sep=',')

# --- Preprocessing ---

# 2. Drop unnecessary columns
unnecessary_columns = ['Timestamp', 'Source IP Address', 'Destination IP Address',
                       'Payload Data', 'Geo-location Data', 'Log Source']
data = data.drop(columns=unnecessary_columns)

# 3. Encoding Fitur Kategorikal
categorical_features = ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Type',
                        'Attack Signature', 'Action Taken', 'User Information',
                        'Device Information', 'Network Segment', 'Malware Indicators',
                        'Alerts/Warnings']

for feature in categorical_features:
    if data[feature].dtype == 'object':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(data[[feature]])
        encoded_df = pd.DataFrame(encoded_features, columns=[f"{feature}_{val}" for val in encoder.categories_[0]])
        data = data.drop(feature, axis=1)
        data = pd.concat([data, encoded_df], axis=1)

# 4. Konversi 'Anomaly Scores' ke Numerik (jika perlu)
if data['Anomaly Scores'].dtype == 'object':
    data['Anomaly Scores'] = pd.to_numeric(data['Anomaly Scores'], errors='coerce')
    data['Anomaly Scores'].fillna(data['Anomaly Scores'].median(), inplace=True)

# 5. Label Encoding 'Severity Level'
label_encoder = LabelEncoder()
data['Severity Level'] = label_encoder.fit_transform(data['Severity Level'])

# --- Akhir Preprocessing ---

# 6. Pisahkan Fitur (X) dan Target (y)
X = data.drop('Severity Level', axis=1)
y = data['Severity Level']

# 7. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning (Opsional) ---
# (Menggunakan RandomizedSearchCV seperti di tutorial)

param_dist = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 20)}

rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_

print('Best hyperparameters:', rand_search.best_params_)

# --- Akhir Hyperparameter Tuning ---

# 8. Inisialisasi dan Latih Model (gunakan best_rf jika tuning dilakukan)
model = best_rf if 'best_rf' in locals() else RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Prediksi dan Evaluasi
y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Visualisasi Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()