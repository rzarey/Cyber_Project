import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Load Data
data_path = 'data-train.csv'
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

# Gunakan OneHotEncoder dengan sparse_output=True
for feature in categorical_features:
    if data[feature].dtype == 'object':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)  # sparse_output=True di sini!
        encoded_features = encoder.fit_transform(data[[feature]])
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=[f"{feature}_{val}" for val in encoder.categories_[0]])  # Konversi ke array agar bisa digabung
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

# 8. Inisialisasi dan Latih Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Prediksi dan Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))