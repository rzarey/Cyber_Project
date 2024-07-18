import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Load Data
data_path = 'data-train.csv'
data = pd.read_csv(data_path)

# --- Preprocessing ---

# 2. Drop unnecessary columns
unnecessary_columns = ['Timestamp', 'Source IP Address', 'Destination IP Address',
                       'Payload Data', 'Geo-location Data', 'Log Source']
data = data.drop(columns=unnecessary_columns)

# 3. Encoding Fitur Kategorikal
categorical_features = ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Type',
                        'Attack Signature', 'Action Taken', 'User Information',
                        'Device Information', 'Network Segment']

for feature in categorical_features:
    if data[feature].dtype == 'object':
        if data[feature].nunique() <= 10:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Changed sparse to sparse_output
            encoded_features = encoder.fit_transform(data[[feature]])
            encoded_df = pd.DataFrame(encoded_features, columns=[f"{feature}_{val}" for val in encoder.categories_[0]])
        else:
            encoder = LabelEncoder()
            data[feature] = encoder.fit_transform(data[feature])
            encoded_df = data[[feature]]
        data = data.drop(feature, axis=1)
        data = pd.concat([data, encoded_df], axis=1)

# 4. Konversi 'Anomaly Scores' ke Numerik (jika perlu)
if data['Anomaly Scores'].dtype == 'object':
    data['Anomaly Scores'] = pd.to_numeric(data['Anomaly Scores'], errors='coerce')
    data['Anomaly Scores'].fillna(data['Anomaly Scores'].mean(), inplace=True)

# 5. Konversi 'Severity Level' ke Numerik (jika perlu)
if data['Severity Level'].dtype == 'object':
    data['Severity Level'] = pd.to_numeric(data['Severity Level'], errors='coerce')
    data['Severity Level'].fillna(data['Severity Level'].mean(), inplace=True)

# --- Akhir Preprocessing ---

# Pastikan tidak ada kolom dengan nilai string yang tersisa
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Kolom {col} masih memiliki tipe data object.")

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
