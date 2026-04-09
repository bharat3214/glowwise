import joblib
import pandas as pd
import numpy as np
import shap

class MLEngine:
    def __init__(self, model_path='models/model.pkl', encoder_path='models/encoders.pkl'):
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoder_path)
        # TreeExplainer calculates SHAP values rapidly for Random Forest
        self.explainer = shap.TreeExplainer(self.model)

    def encode_features(self, user_features: dict) -> pd.DataFrame:
        df = pd.DataFrame([user_features])
        df_encoded = pd.DataFrame()
        for col in df.columns:
            if col in self.encoders:
                le = self.encoders[col]
                try:
                    df_encoded[col] = le.transform(df[col].astype(str))
                except ValueError:
                    df_encoded[col] = [0]
        return df_encoded

    def predict_routine(self, user_features: dict) -> str:
        df_encoded = self.encode_features(user_features)
        pred_encoded = self.model.predict(df_encoded)
        routine = self.encoders['target_routine'].inverse_transform(pred_encoded)[0]
        return routine
        
    def explain_prediction(self, user_features: dict) -> dict:
        df_encoded = self.encode_features(user_features)
        pred_encoded = self.model.predict(df_encoded)[0]
        
        shap_values = self.explainer.shap_values(df_encoded)
        
        # Scikit-Learn RF Classifier SHAP returns a list of matrices per class
        if isinstance(shap_values, list):
            vals = shap_values[pred_encoded][0]
        elif len(np.array(shap_values).shape) == 3:
            vals = np.array(shap_values)[0, :, pred_encoded]
        else:
            vals = shap_values[0]
            
        feature_names = df_encoded.columns.tolist()
        
        contributions = {}
        for feature, val in zip(feature_names, vals):
            # Only track features with non-negligible impact to keep UI clean
            if abs(val) > 0.001:
                label = f"{feature} ({user_features[feature]})"
                contributions[label] = float(val)
                
        # Sort by absolute impact
        sorted_contribs = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
        return sorted_contribs
