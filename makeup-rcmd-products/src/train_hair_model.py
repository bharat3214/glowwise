import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_and_save_hair_model():
    dataset_path = 'data/haircare_dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found. Please run generate_hair_data.py first.")

    df = pd.read_csv(dataset_path)

    # Features and Target
    X = df.drop(columns=['Routine'])
    y = df['Routine']

    encoders = {}
    X_encoded = pd.DataFrame()

    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y.astype(str))
    encoders['target_routine'] = le_target

    model = RandomForestClassifier(n_estimators=100, random_state=123)
    model.fit(X_encoded, y_encoded)
    
    # Simple training accuracy
    acc = model.score(X_encoded, y_encoded)
    print(f"Haircare Training Accuracy: {acc * 100:.2f}%")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/hair_model.pkl')
    joblib.dump(encoders, 'models/hair_encoders.pkl')
    print("Haircare model and encoders saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_save_hair_model()
