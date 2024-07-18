import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import randint
import os
import pickle
import time

class ModelTrainer:
    def __init__(self, data_path, unnecessary_columns, categorical_features):
        self.data_path = data_path
        self.unnecessary_columns = unnecessary_columns
        self.categorical_features = categorical_features

    def load_data(self):
        self.data = pd.read_csv(self.data_path, sep=',')
        self.data = self.data.drop(columns=self.unnecessary_columns)

    def preprocess_data(self):
        self.encode_categorical_features()
        self.convert_anomaly_scores()
        self.label_encode_severity_level()

    def encode_categorical_features(self):
        for feature in self.categorical_features:
            if feature in self.data.columns and self.data[feature].dtype == 'object':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_features = encoder.fit_transform(self.data[[feature]])
                encoded_df = pd.DataFrame(encoded_features, columns=[f"{feature}_{val}" for val in encoder.categories_[0]])
                self.data = self.data.drop(feature, axis=1)
                self.data = pd.concat([self.data, encoded_df], axis=1)

    def convert_anomaly_scores(self):
        if self.data['Anomaly Scores'].dtype == 'object':
            self.data['Anomaly Scores'] = pd.to_numeric(self.data['Anomaly Scores'], errors='coerce')
            self.data['Anomaly Scores'].fillna(self.data['Anomaly Scores'].median(), inplace=True)

    def label_encode_severity_level(self):
        label_encoder = LabelEncoder()
        self.data['Severity Level'] = label_encoder.fit_transform(self.data['Severity Level'])

    def split_data(self):
        X = self.data.drop('Severity Level', axis=1)
        y = self.data['Severity Level']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
        rf = RandomForestClassifier()
        rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
        start_time = time.time()
        rand_search.fit(self.X_train, self.y_train)
        self.training_time = time.time() - start_time
        self.best_rf = rand_search.best_estimator_
        print('Best hyperparameters:', rand_search.best_params_)

    def evaluate_model(self):
        self.y_pred = self.best_rf.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, self.y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, self.y_pred))
        cm = confusion_matrix(self.y_test, self.y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.best_rf, file)
        print(f"Model saved to {model_path}")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
        self.save_model(model_path)
        print(f"Training time: {self.training_time:.2f} seconds")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'data-train.csv')
    unnecessary_columns = ['Timestamp', 'Source IP Address', 'Destination IP Address', 'Payload Data', 'Geo-location Data', 'Log Source']
    categorical_features = ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Type', 'Attack Signature', 'Action Taken', 'User Information', 'Device Information', 'Network Segment', 'Malware Indicators', 'Alerts/Warnings']

    trainer = ModelTrainer(data_path, unnecessary_columns, categorical_features)
    trainer.run()
